"""过敏原检测套餐规划测试（不依赖 FastAPI，走 db + review + allergen_panel 全链路）。

覆盖：
- 仅允许选择一条时效正常记录，读取其当前配方（两版取新版）最终过敏原；
- 「套餐名：价格：过敏原列表」逐行解析：三段格式、价格精确小数、空过敏原、
  注释空行、套餐名去空与重名整批拒绝（逐行给出问题）；
- 子集动态规划：最低总价覆盖、同价按套餐名序列字典序取唯一、并列数，
  对照暴力枚举；**零价冗余套餐**（覆盖至少一个必须项但加入后不新增覆盖）
  作为不同最低价组合计入并列，正价冗余不进入最低价；字典序最小方案允许
  空覆盖贡献且累计覆盖保持正确；
- 精确有理数（0.1 + 0.2 == 0.3）；
- 无法覆盖时列出缺口；覆盖贡献与累计覆盖；额外过敏原与无用套餐不参与组合；
- 缺失 / 非法 ID / 需复核记录拒绝并说明；全程只读、不写库。
"""
import os
import tempfile
import unittest
from fractions import Fraction
from itertools import combinations

_DB_DIR = tempfile.mkdtemp(prefix="allergen-panel-")
os.environ["DB_PATH"] = os.path.join(_DB_DIR, "panel.db")

from app import db  # noqa: E402
from app.allergen_panel import (PanelError, build_panel_preview,  # noqa: E402
                                list_panel_records, parse_packages,
                                solve_panel)
from app.recheck import build_recheck  # noqa: E402
from app.review import analyze  # noqa: E402

DICT_TEXT = """\
小麦粉：麸质谷物
乳清粉：乳
乳糖：乳
奶粉：乳
可可脂：
大豆卵磷脂：大豆
鸡蛋：蛋类
椰子粉：坚果
芝麻：芝麻
食盐：
"""
TEXT_MILK = "配料表：小麦粉，白砂糖，巧克力酱（可可脂，奶粉（乳清粉，乳糖），大豆卵磷脂），鸡蛋，食盐"
TEXT_B = "配料表：小麦粉，椰子粉，鸡蛋，芝麻，食盐"


def make_review(text_a, text_b=None, name="套餐测试", with_fingerprint=True):
    result, fp = analyze(text_a=text_a, text_b=text_b, label_a="现版配方",
                         label_b="新版配方", declared=[])
    return db.create_review(
        name=name, declared=[], text_a=text_a, text_b=text_b,
        label_a="现版配方", label_b="新版配方", result=result,
        dict_fingerprint=fp if with_fingerprint else None)


def brute_force(target, packages):
    """暴力枚举全部套餐子集，返回 (最低总价 Fraction|None, 名称键, 方案数)。"""
    required = set(target)
    useful = [p for p in packages if set(p["allergens"]) & required]
    best, best_key, ways = None, None, 0
    for r in range(len(useful) + 1):
        for combo in combinations(useful, r):
            covered = set()
            for p in combo:
                covered |= set(p["allergens"]) & required
            if not required <= covered:
                continue
            cost = sum((p["price"] for p in combo), Fraction(0))
            key = tuple(sorted(p["name"] for p in combo))
            if best is None or cost < best:
                best, best_key, ways = cost, key, 1
            elif cost == best:
                ways += 1
                if key < best_key:
                    best_key = key
    return best, best_key, ways


class ParseTest(unittest.TestCase):
    """套餐行解析：格式、精确价格、重名、注释。"""

    def test_basic_parse(self):
        pkgs = parse_packages(
            "# 注释\n\n基础套餐：9.90：麸质谷物，蛋类\n"
            "空包:0:\n乳品：10：乳；坚果")
        self.assertEqual([p["name"] for p in pkgs], ["基础套餐", "空包", "乳品"])
        self.assertEqual(pkgs[0]["price"], Fraction("9.90"))
        self.assertEqual(pkgs[0]["allergens"], ["蛋类", "麸质谷物"])
        self.assertEqual(pkgs[1]["allergens"], [])
        self.assertEqual(pkgs[2]["allergens"], ["乳", "坚果"])

    def test_format_and_price_and_duplicate(self):
        with self.assertRaises(PanelError) as ctx:
            parse_packages("只有名\n  ：1：乳\nA：1：乳\nA：2：蛋类")
        types = [(p["line"], p["type"]) for p in ctx.exception.problems]
        self.assertEqual(types, [(1, "format"), (2, "empty_name"),
                                 (4, "duplicate_name")])
        with self.assertRaises(PanelError) as ctx:
            parse_packages("A：abc：乳")
        self.assertEqual(ctx.exception.problems[0]["type"], "bad_price")
        with self.assertRaises(PanelError) as ctx:
            parse_packages("A：-1：乳")
        self.assertEqual(ctx.exception.problems[0]["type"], "bad_price")
        with self.assertRaises(PanelError) as ctx:
            parse_packages("A：1：2：乳")  # 三段以上
        self.assertEqual(ctx.exception.problems[0]["type"], "format")

    def test_exponent_price_rejected(self):
        with self.assertRaises(PanelError):
            parse_packages("A：1e3：乳")


class DPTest(unittest.TestCase):
    """子集动态规划纯计算：最低总价、并列字典序（对照暴力枚举）、缺口。"""

    def _pkgs(self, spec):
        return [{"name": n, "price": Fraction(str(price)), "allergens": list(al)}
                for n, price, al in spec]

    def test_min_cost_and_contribution(self):
        target = ["乳", "大豆", "蛋类", "麸质谷物"]
        pkgs = self._pkgs([
            ("基础", "9.9", ["麸质谷物", "蛋类"]),
            ("乳品", "10", ["乳"]),
            ("全包", "30", ["乳", "大豆", "蛋类", "麸质谷物"]),
            ("大豆加购", "5", ["大豆"]),
            ("零价空包", "0", []),
        ])
        r = solve_panel(target, pkgs)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["total_price"], 24.9)
        self.assertEqual([s["name"] for s in r["selections"]],
                         ["乳品", "基础", "大豆加购"])
        # 覆盖贡献与累计覆盖按套餐名序列展示
        self.assertEqual(r["selections"][0]["contribution"], ["乳"])
        self.assertEqual(r["selections"][0]["cumulative_covered"], ["乳"])
        self.assertEqual(r["selections"][1]["contribution"], ["蛋类", "麸质谷物"])
        self.assertEqual(r["selections"][2]["contribution"], ["大豆"])
        self.assertEqual(r["selections"][2]["cumulative_covered"],
                         ["乳", "大豆", "蛋类", "麸质谷物"])
        # 无必须项覆盖的套餐不参与组合
        self.assertEqual([p["name"] for p in r["unused_packages"]], ["零价空包"])

    def test_tie_break_and_count(self):
        # 0.1 + 0.2 与 0.3 同价：(X,Y) 与 (Z) 并列，字典序取 (X,Y)
        target = ["乳", "大豆", "蛋类", "麸质谷物"]
        pkgs = self._pkgs([
            ("X", "0.1", ["乳"]),
            ("Y", "0.2", ["大豆", "蛋类", "麸质谷物"]),
            ("Z", "0.3", ["乳", "大豆", "蛋类", "麸质谷物"]),
        ])
        r = solve_panel(target, pkgs)
        self.assertEqual(r["total_price"], 0.3)
        self.assertEqual(r["optimal_count"], 2)
        self.assertEqual([s["name"] for s in r["selections"]], ["X", "Y"])

    def test_zero_price_redundant_counted_as_tie(self):
        # 「大豆」仅正价的全包能覆盖，故全包（1 元）必选；两个 0 元套餐 A、B
        # 覆盖的乳 / 蛋类已被全包覆盖，加入后不新增覆盖，但每个都构成不同的
        # 最低价组合：{全包}、{A,全包}、{B,全包}、{A,B,全包} 共 4 个并列。
        target = ["乳", "大豆", "蛋类"]
        pkgs = self._pkgs([
            ("全包", "1", ["乳", "大豆", "蛋类"]),
            ("A", "0", ["乳"]),
            ("B", "0", ["蛋类"]),
        ])
        r = solve_panel(target, pkgs)
        bc, bkey, bways = brute_force(target, pkgs)
        self.assertTrue(r["feasible"])
        self.assertEqual(Fraction(str(r["total_price"])), bc)
        self.assertEqual(r["optimal_count"], 4)
        self.assertEqual(r["optimal_count"], bways)
        # 字典序最小方案为 (A, B, 全包)：A、B 先各贡献一项，全包只新覆盖大豆
        self.assertEqual([s["name"] for s in r["selections"]],
                         ["A", "B", "全包"])
        self.assertEqual(tuple(s["name"] for s in r["selections"]), bkey)
        by_name = {s["name"]: s for s in r["selections"]}
        self.assertEqual(by_name["A"]["contribution"], ["乳"])
        self.assertEqual(by_name["A"]["cumulative_covered"], ["乳"])
        self.assertEqual(by_name["B"]["contribution"], ["蛋类"])
        self.assertEqual(by_name["B"]["cumulative_covered"], ["乳", "蛋类"])
        self.assertEqual(by_name["全包"]["contribution"], ["大豆"])
        self.assertEqual(by_name["全包"]["cumulative_covered"],
                         ["乳", "大豆", "蛋类"])

    def test_zero_price_empty_contribution_preserves_coverage(self):
        # M 正价且唯一覆盖大豆（必选），Z 正价覆盖蛋（必选）；0 元套餐 N 覆盖
        # 的乳在展示方案中已由名序更靠前的 M 覆盖 → N 的覆盖贡献为空、累计覆盖
        # 不变化，但 {M,Z} 与 {M,N,Z} 同为 3 元，并列数为 2。
        target = ["乳", "大豆", "蛋"]
        pkgs = self._pkgs([
            ("M正", "1", ["乳", "大豆"]),
            ("N零", "0", ["乳"]),
            ("Z蛋", "2", ["蛋"]),
        ])
        r = solve_panel(target, pkgs)
        bc, bkey, bways = brute_force(target, pkgs)
        self.assertTrue(r["feasible"])
        self.assertEqual(Fraction(str(r["total_price"])), 3)
        self.assertEqual(Fraction(str(r["total_price"])), bc)
        self.assertEqual(r["optimal_count"], 2)
        self.assertEqual(r["optimal_count"], bways)
        # (M正, N零, Z蛋) 字典序小于 (M正, Z蛋)
        self.assertEqual([s["name"] for s in r["selections"]],
                         ["M正", "N零", "Z蛋"])
        self.assertEqual(tuple(s["name"] for s in r["selections"]), bkey)
        by_name = {s["name"]: s for s in r["selections"]}
        self.assertEqual(by_name["M正"]["contribution"], ["乳", "大豆"])
        self.assertEqual(by_name["N零"]["contribution"], [])
        self.assertEqual(by_name["N零"]["contribution_count"], 0)
        # 空贡献：累计覆盖与前一行完全一致，不重复计入
        self.assertEqual(by_name["N零"]["cumulative_covered"], ["乳", "大豆"])
        self.assertEqual(by_name["Z蛋"]["contribution"], ["蛋"])
        self.assertEqual(by_name["Z蛋"]["cumulative_covered"],
                         ["乳", "大豆", "蛋"])

    def test_positive_price_redundant_never_optimal(self):
        # 全包 0 元即可全覆盖：正价冗余套餐（即使只多 0.1 元）不得进入最低价，
        # 并列组合数仍为 1；正价精确有理数（0.1）与零价不会误判同价。
        target = ["乳", "蛋类"]
        pkgs = self._pkgs([
            ("全包", "0", ["乳", "蛋类"]),
            ("贵包", "0.1", ["乳"]),
            ("蛋加购", "1.5", ["蛋类"]),
        ])
        r = solve_panel(target, pkgs)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["total_price"], 0)
        self.assertEqual(r["optimal_count"], 1)
        self.assertEqual([s["name"] for s in r["selections"]], ["全包"])

    def test_matches_brute_force(self):
        import random
        random.seed(7)
        target = ["A", "B", "C", "D"]
        universe = ["A", "B", "C", "D", "X", "Y"]
        for trial in range(60):
            spec = []
            for i in range(random.randint(1, 7)):
                # 含 0 价：覆盖至少一个必须项的零价冗余套餐必须计入并列，
                # 对照暴力枚举验证 DP 的方案数、总价与字典序唯一方案
                spec.append((f"P{i}", random.choice(["1", "2", "3", "0.5", "0"]),
                             random.sample(universe, random.randint(0, 4))))
            pkgs = self._pkgs(spec)
            r = solve_panel(target, pkgs)
            bc, bkey, bways = brute_force(target, pkgs)
            if bc is None:
                self.assertFalse(r["feasible"])
                # 缺口 = 没有任何套餐覆盖的必查项
                reach = set().union(*[set(p["allergens"]) for p in pkgs])
                self.assertEqual(r["uncovered"],
                                 sorted(set(target) - reach))
            else:
                self.assertTrue(r["feasible"])
                self.assertEqual(Fraction(str(r["total_price"])), bc)
                self.assertEqual(r["optimal_count"], bways)
                self.assertEqual(tuple(s["name"] for s in r["selections"]), bkey)

    def test_empty_target(self):
        r = solve_panel([], self._pkgs([("A", "1", ["乳"])]))
        self.assertTrue(r["feasible"])
        self.assertEqual(r["total_price"], 0)
        self.assertEqual(r["selections"], [])
        self.assertEqual(r["optimal_count"], 1)

    def test_extra_allergens_reported(self):
        pkgs = self._pkgs([("A", "5", ["乳", "鱼类"])])
        r = solve_panel(["乳"], pkgs)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["selections"][0]["extra_allergens"], ["鱼类"])
        self.assertEqual(r["packages"][0]["covers_required"], ["乳"])


class PreviewFlowTest(unittest.TestCase):
    """全链路：记录资格、两版取 B、缺口、需复核 / 缺失拒绝、只读。"""

    def setUp(self):
        db.DB_PATH = os.path.join(
            _DB_DIR, f"panel_{self.id().rsplit('.', 1)[-1]}.db")
        db.init_db()
        db.bulk_import(DICT_TEXT)

    def _fp(self, rid):
        with db._connect() as conn:
            return conn.execute(
                "SELECT dict_fingerprint FROM reviews WHERE id=?",
                (rid,)).fetchone()[0]

    def test_required_allergens_from_current_version(self):
        rid = make_review(TEXT_MILK, TEXT_B, name="两版")
        records = {r["id"]: r for r in list_panel_records()}
        self.assertEqual(records[rid]["source_version"], "新版配方")
        r = build_panel_preview(rid, "全包：99：麸质谷物，坚果，蛋类，芝麻")
        self.assertEqual(r["required_allergens"],
                         ["坚果", "芝麻", "蛋类", "麸质谷物"])
        self.assertTrue(r["feasible"])

    def test_feasible_plan_and_gap(self):
        rid = make_review(TEXT_MILK)
        r = build_panel_preview(rid, "乳包：10：乳\n基础：9.9：麸质谷物，蛋类")
        self.assertFalse(r["feasible"])
        self.assertEqual(r["total_price"], None)
        self.assertEqual(r["uncovered"], ["大豆"])
        r = build_panel_preview(
            rid, "乳包：10：乳\n基础：9.9：麸质谷物，蛋类\n大豆：5：大豆")
        self.assertTrue(r["feasible"])
        self.assertEqual(r["total_price"], 24.9)
        self.assertEqual(r["optimal_count"], 1)

    def test_empty_packages_rejected(self):
        rid = make_review(TEXT_MILK)
        with self.assertRaises(PanelError):
            build_panel_preview(rid, "# 只有注释\n")

    def test_missing_and_invalid_record_blocked(self):
        with self.assertRaises(PanelError) as ctx:
            build_panel_preview(999999, "A：1：乳")
        self.assertEqual(ctx.exception.problems[0]["reason"], "missing")
        with self.assertRaises(PanelError) as ctx:
            build_panel_preview("x", "A：1：乳")
        self.assertEqual(ctx.exception.problems[0]["reason"], "invalid")

    def test_stale_record_blocked_until_rechecked(self):
        rid = make_review(TEXT_MILK)
        db.upsert_entry("乳清粉", ["乳", "花生"])  # 映射变更 → 需复核
        self.assertTrue(db.get_review(rid)["needs_recheck"])
        with self.assertRaises(PanelError) as ctx:
            build_panel_preview(rid, "A：1：乳")
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "stale")
        self.assertIn("需复核", p["message"])
        records = {r["id"]: r for r in list_panel_records()}
        self.assertFalse(records[rid]["eligible"])
        # 复核后恢复
        outcome = build_recheck(db.get_review(rid))
        db.update_review_result(rid, outcome["new_result"],
                                outcome["fingerprint"])
        self.assertFalse(db.get_review(rid)["needs_recheck"])
        r = build_panel_preview(rid, "全包：1：乳，花生，大豆，蛋类，麸质谷物")
        self.assertTrue(r["feasible"])

    def test_legacy_record_blocked(self):
        rid = make_review(TEXT_MILK, with_fingerprint=False)
        with self.assertRaises(PanelError) as ctx:
            build_panel_preview(rid, "A：1：乳")
        self.assertEqual(ctx.exception.problems[0]["reason"], "legacy")

    def test_deleted_record_blocked(self):
        rid = make_review(TEXT_MILK)
        db.delete_review(rid)
        with self.assertRaises(PanelError) as ctx:
            build_panel_preview(rid, "A：1：乳")
        self.assertEqual(ctx.exception.problems[0]["reason"], "missing")

    def test_readonly(self):
        rid = make_review(TEXT_MILK)
        before = db.get_review(rid)
        count = len(db.list_reviews())
        build_panel_preview(rid, "全包：99：麸质谷物，乳，大豆，蛋类")
        build_panel_preview(rid, "乳：1：乳")  # 有缺口的无解计算也不写库
        self.assertEqual(db.get_review(rid), before)
        self.assertEqual(self._fp(rid), before["dict_fingerprint"])
        self.assertEqual(len(db.list_reviews()), count)


if __name__ == "__main__":
    unittest.main()
