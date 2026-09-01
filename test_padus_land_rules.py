"""Focused tests for configurable PAD-US land classification."""

from pathlib import Path
import unittest

from padus_land_rules import (
    NASA_LOCAL_MANAGER,
    RuleConfigError,
    build_rule_context,
    load_rule_config,
    parse_rule_config,
)


RULE_PATH = Path("config/padus_land_rules.rules")
ALLOWED_VALUES = {
    "Own_Type": {
        "DESG",
        "DIST",
        "FED",
        "JNT",
        "LOC",
        "NGO",
        "PVT",
        "STAT",
        "TERR",
        "TRIB",
        "UNK",
    },
    "Mang_Type": {
        "DESG",
        "DIST",
        "FED",
        "JNT",
        "LOC",
        "NGO",
        "PVT",
        "STAT",
        "TERR",
        "TRIB",
        "UNK",
    },
    "Des_Tp": {
        "MIL",
        "PAGR",
        "PCON",
        "PFOR",
        "PHCA",
        "POTH",
        "PPRK",
        "PRAN",
        "PREC",
        "WPA",
    },
    "Pub_Access": {"OA", "RA", "XA", "UK"},
    "manager": {"NASA", "DOE", "DOD"},
}


class PadusLandRulesTest(unittest.TestCase):
    """Verify parsing, validation, and representative policy cases."""

    @classmethod
    def setUpClass(cls) -> None:
        _, cls.rules = load_rule_config(RULE_PATH)
        cls.rules.validate(ALLOWED_VALUES)

    def classify(
        self,
        *,
        own_type: str,
        mang_type: str,
        des_type: str = "WPA",
        mang_name: str = "BLM",
        loc_mang: str = "Bureau of Land Management",
        pub_access: str = "UK",
    ) -> dict[str, bool]:
        """Classify one representative PAD-US attribute combination.

        Args:
            own_type: PAD-US owner-type code.
            mang_type: PAD-US manager-type code.
            des_type: PAD-US designation-type code.
            mang_name: PAD-US national manager code.
            loc_mang: PAD-US local manager value.
            pub_access: PAD-US public-access code.

        Returns:
            Boolean public-land and public-access classifications.
        """
        context = build_rule_context(
            {
                "Own_Type": own_type,
                "Mang_Type": mang_type,
                "Des_Tp": des_type,
                "Pub_Access": pub_access,
                "Mang_Name": mang_name,
                "Loc_Mang": loc_mang,
            }
        )
        return self.rules.evaluate(context)

    def test_public_owner_types_are_public_land(self) -> None:
        for own_type in ("FED", "JNT", "LOC", "DIST", "STAT"):
            with self.subTest(own_type=own_type):
                result = self.classify(own_type=own_type, mang_type="PVT")
                self.assertTrue(result["public_land"])
                self.assertTrue(result["public_access"])

        for own_type in ("TERR", "TRIB"):
            with self.subTest(own_type=own_type):
                result = self.classify(own_type=own_type, mang_type="FED")
                self.assertFalse(result["public_land"])
                self.assertFalse(result["public_access"])

    def test_fallback_owner_requires_public_manager(self) -> None:
        for own_type in ("DESG", "NGO", "PVT", "UNK"):
            for mang_type in ("FED", "LOC", "DIST", "STAT"):
                with self.subTest(own_type=own_type, mang_type=mang_type):
                    self.assertTrue(
                        self.classify(
                            own_type=own_type,
                            mang_type=mang_type,
                        )["public_land"]
                    )

        excluded = self.classify(own_type="PVT", mang_type="NGO")
        self.assertFalse(excluded["public_land"])
        self.assertFalse(excluded["public_access"])

    def test_restricted_and_unknown_access_can_be_public(self) -> None:
        for pub_access in ("RA", "UK"):
            with self.subTest(pub_access=pub_access):
                result = self.classify(
                    own_type="FED",
                    mang_type="FED",
                    pub_access=pub_access,
                )
                self.assertTrue(result["public_access"])

    def test_nonopen_access_excludes_private_owners(self) -> None:
        result = self.classify(
            own_type="PVT",
            mang_type="FED",
            pub_access="RA",
        )

        self.assertTrue(result["public_land"])
        self.assertFalse(result["public_access"])

    def test_nonopen_access_excludes_configured_designations(self) -> None:
        for des_type in (
            "MIL",
            "PAGR",
            "PCON",
            "PFOR",
            "PHCA",
            "POTH",
            "PPRK",
            "PRAN",
            "PREC",
        ):
            with self.subTest(des_type=des_type):
                result = self.classify(
                    own_type="FED",
                    mang_type="FED",
                    des_type=des_type,
                )
                self.assertTrue(result["public_land"])
                self.assertFalse(result["public_access"])

        self.assertTrue(
            self.classify(
                own_type="FED",
                mang_type="FED",
                des_type="WPA",
            )["public_access"]
        )

    def test_nonopen_access_excludes_configured_managers(self) -> None:
        nasa = self.classify(
            own_type="FED",
            mang_type="FED",
            mang_name="OTHF",
            loc_mang=NASA_LOCAL_MANAGER,
        )
        self.assertTrue(nasa["public_land"])
        self.assertFalse(nasa["public_access"])

        doe = self.classify(
            own_type="FED",
            mang_type="FED",
            mang_name="DOE",
        )
        self.assertTrue(doe["public_land"])
        self.assertFalse(doe["public_access"])

        similar_name = self.classify(
            own_type="FED",
            mang_type="FED",
            mang_name="OTHF",
            loc_mang="NASA Recreation Partnership",
        )
        self.assertTrue(similar_name["public_access"])

        dod = self.classify(
            own_type="FED",
            mang_type="FED",
            des_type="WPA",
            mang_name="DOD",
        )
        self.assertFalse(dod["public_access"])

    def test_open_access_overrides_secondary_exclusions(self) -> None:
        result = self.classify(
            own_type="PVT",
            mang_type="FED",
            des_type="MIL",
            mang_name="DOD",
            pub_access="OA",
        )

        self.assertTrue(result["public_land"])
        self.assertTrue(result["public_access"])

    def test_closed_access_is_not_public_access(self) -> None:
        result = self.classify(
            own_type="FED",
            mang_type="FED",
            pub_access="XA",
        )

        self.assertTrue(result["public_land"])
        self.assertFalse(result["public_access"])

    def test_invalid_syntax_field_and_code_fail_clearly(self) -> None:
        with self.assertRaisesRegex(RuleConfigError, "Expected field"):
            parse_rule_config("public_land == Own_Type in {FED}")

        unknown_field = parse_rule_config(
            "public_land = Owner in {FED} public_access = public_land"
        )
        with self.assertRaisesRegex(RuleConfigError, "unsupported field"):
            unknown_field.validate(ALLOWED_VALUES)

        unknown_code = parse_rule_config(
            "public_land = Own_Type in {FEDERAL} "
            "public_access = public_land"
        )
        with self.assertRaisesRegex(RuleConfigError, "invalid Own_Type value"):
            unknown_code.validate(ALLOWED_VALUES)


if __name__ == "__main__":
    unittest.main()
