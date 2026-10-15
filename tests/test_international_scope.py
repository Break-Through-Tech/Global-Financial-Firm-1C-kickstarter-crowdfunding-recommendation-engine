import unittest
import pandas as pd
from utils.international_scope import (
    classify_location, classify_locations, international_mask,
    align_international_to_global_split, location_scope_audit,
)
from utils.us_scope import us_state_proxy_mask


class InternationalScopeTests(unittest.TestCase):
    def test_explicit_named_international(self):
        cases = {
            "Berlin, Germany": "Germany",
            "Toronto, Canada": "Canada",
            "Taipei, Taiwan": "Taiwan, Province of China",
            "Sydney, Australia": "Australia",
            "London, UK": "United Kingdom",
            "Seoul, South Korea": "Korea, Republic of",
            "Reykjavik, Iceland": "Iceland",
            "Montréal, Canada": "Canada",
        }
        for location, expected_country in cases.items():
            with self.subTest(location=location):
                scope, country, evidence = classify_location(location)
                self.assertEqual(scope, "explicit_non_us")
                self.assertEqual(country, expected_country)
                self.assertEqual(evidence, "explicit_country_or_region_name")

    def test_audited_historical_and_regional_names(self):
        # Location evidence comes from the final explicit name, not the city.
        cases = {
            "Istanbul, Turkey": "Türkiye",
            "Nev�ā�ehir, Turkey": "Türkiye",
            "Gaza City, Palestinian Territories": "Palestine, State of",
            "Tripoli, Libyan Arab Jamahiriya": "Libya",
            "Skopje, Macedonia": "North Macedonia",
            "Stanley, Falkland Islands": "Falkland Islands (Malvinas)",
            "Palikir, Micronesia": "Micronesia, Federated States of",
        }
        for location, expected in cases.items():
            with self.subTest(location=location):
                scope, country, evidence = classify_location(location)
                self.assertEqual(scope, "explicit_non_us")
                self.assertEqual(country, expected)
                self.assertEqual(evidence, "explicit_country_or_region_name")

    def test_truncated_and_ambiguous_names_stay_unknown(self):
        cases = ["Ciudad Autonoma De Buenos Aires, Argent",
                 "Santiago De Los Caballeros, Dominican Re",
                 "Tbilisi, Georgia", "Kinshasa, Congo",
                 "Maar", None]
        for location in cases:
            with self.subTest(location=location):
                self.assertEqual(classify_location(location)[0], "unknown")

    def test_territory_aliases_precede_country_assignment(self):
        for location in ("St Croix, Virgin Islands, U.S.",
                         "Cruz Bay, Virgin Islands, U.S.",
                         "Virgin Islands of the United States",
                         "City, Virgin Islands of the United States",
                         "City, United States Minor Outlying Islands"):
            with self.subTest(location=location):
                self.assertEqual(classify_location(location)[0], "us_territory")

    def test_comma_containing_official_country_names(self):
        for location, expected in {
            "Seoul, Korea, Republic of": "Korea, Republic of",
            "La Paz, Bolivia, Plurinational State of": "Bolivia, Plurinational State of",
            "Kinshasa, Congo, The Democratic Republic of the": "Congo, The Democratic Republic of the",
        }.items():
            with self.subTest(location=location):
                self.assertEqual(classify_location(location)[:2], ("explicit_non_us", expected))

    def test_frozen_proxy_agrees_rowwise_and_categories_partition(self):
        locations = ["Charlottesville, Virginia", "San Francisco, CA", "CA",
                     "Toronto, ON", "Tbilisi, Georgia", "Kinshasa, Congo",
                     "St Croix, Virgin Islands, U.S.", "Berlin, Germany", None,
                     "Boston, ma"]
        frame = pd.DataFrame({"location": locations})
        geo, overview, _, _ = location_scope_audit(frame)
        self.assertEqual(geo.scope.tolist(), ["explicit_us", "us_candidate", "unknown",
                         "unknown", "unknown", "unknown", "us_territory",
                         "explicit_non_us", "unknown", "us_candidate"])
        self.assertEqual(geo.iloc[0].evidence, "full_state_name_outside_proxy")
        pd.testing.assert_series_equal(geo.scope.eq("us_candidate"),
                                      us_state_proxy_mask(frame), check_names=False)
        scopes = ["explicit_us", "us_candidate", "unknown", "us_territory", "explicit_non_us"]
        self.assertTrue(pd.concat([geo.scope.eq(s) for s in scopes], axis=1).sum(axis=1).eq(1).all())
        self.assertEqual(overview.n.sum(), len(frame))

    def test_us_does_not_become_international(self):
        for location in ("San Francisco, CA", "Phoenix, AZ", "Portland, USA",
                         "Portland, United States", "San Juan, Puerto Rico"):
            with self.subTest(location=location):
                self.assertNotEqual(classify_location(location)[0], "explicit_non_us")

    def test_ambiguous_and_missing_never_become_international(self):
        for location in (None, "", "  ", "Toronto, ON", "Berlin, BE",
                         "London", "Tbilisi, Georgia", "St Helier, Jersey",
                         "City, CA", "Korea"):
            with self.subTest(location=location):
                self.assertEqual(classify_location(location)[0], "unknown" if location not in ("City, CA",) else "us_candidate")

    def test_masks_disjoint_and_unknown_outside(self):
        df = pd.DataFrame({"location": ["Berlin, Germany", "Paris, France", "Seattle, WA",
                                          None, "London, ON", "Portland, USA"]})
        int_mask = international_mask(df)
        us_mask = us_state_proxy_mask(df)
        self.assertEqual(int_mask.tolist(), [True, True, False, False, False, False])
        self.assertFalse((int_mask & us_mask).any())
        geo, overview, countries, unknowns = location_scope_audit(df)
        self.assertEqual(len(geo), len(df))
        self.assertEqual(countries["n"].sum(), 2)
        self.assertEqual(geo["scope"].eq("unknown").sum(), 2)

    def test_split_reuses_global_membership_not_re_randomized(self):
        clean = pd.DataFrame({"project id":[1,2,3,4],
                "location":["Berlin, Germany"]*4,
                "status":["failed","successful","live","failed"]})
        train = pd.DataFrame({"project id":[4,3,1], "target":[0,0,0]})
        test = pd.DataFrame({"project id":[2], "target":[1]})
        tr, te = align_international_to_global_split(clean, train, test)
        self.assertEqual(tr["project id"].tolist(),[4,1])
        self.assertEqual(te["project id"].tolist(),[2])

    def test_refuses_unknown_in_international_cohort(self):
        clean = pd.DataFrame({"project id":[1], "location":["London, ON"],
                              "status":["successful"]})
        ids = pd.DataFrame({"project id":[1],"target":[1]})
        with self.assertRaises(ValueError):
            align_international_to_global_split(clean, ids, ids.iloc[0:0])


if __name__ == "__main__":
    unittest.main()
