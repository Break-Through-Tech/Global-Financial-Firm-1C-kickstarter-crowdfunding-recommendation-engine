"""Conservative location-scope evidence for Kickstarter's historical text field.

A valid non-U.S. membership requires an explicit trailing named country/region.
Never treat NOT matching the U.S. state proxy as evidence of another country.
The ISO country/region name snapshot below is deliberately vendored (no added
runtime dependencies). It does NOT prove Kickstarter account country, billing
country, currency, or that a scraped location was unchanged at launch.
"""

from pathlib import Path
import re
import unicodedata
import pandas as pd
from utils.us_scope import US_STATE_CODES, us_state_proxy_mask, align_to_global_split

# ISO-3166 common/official English geographic names captured from pycountry 24.6.
# They are linguistic identifiers, not assertions about political status.
COUNTRY_NAMES = {
    'Afghanistan': 'Afghanistan',
    'Albania': 'Albania',
    'Algeria': 'Algeria',
    'American Samoa': 'American Samoa',
    'Andorra': 'Andorra',
    'Angola': 'Angola',
    'Anguilla': 'Anguilla',
    'Antarctica': 'Antarctica',
    'Antigua and Barbuda': 'Antigua and Barbuda',
    'Arab Republic of Egypt': 'Egypt',
    'Argentina': 'Argentina',
    'Argentine Republic': 'Argentina',
    'Armenia': 'Armenia',
    'Aruba': 'Aruba',
    'Australia': 'Australia',
    'Austria': 'Austria',
    'Azerbaijan': 'Azerbaijan',
    'Bahamas': 'Bahamas',
    'Bahrain': 'Bahrain',
    'Bangladesh': 'Bangladesh',
    'Barbados': 'Barbados',
    'Belarus': 'Belarus',
    'Belgium': 'Belgium',
    'Belize': 'Belize',
    'Benin': 'Benin',
    'Bermuda': 'Bermuda',
    'Bhutan': 'Bhutan',
    'Bolivarian Republic of Venezuela': 'Venezuela, Bolivarian Republic of',
    'Bolivia': 'Bolivia, Plurinational State of',
    'Bolivia, Plurinational State of': 'Bolivia, Plurinational State of',
    'Bonaire, Sint Eustatius and Saba': 'Bonaire, Sint Eustatius and Saba',
    'Bosnia and Herzegovina': 'Bosnia and Herzegovina',
    'Botswana': 'Botswana',
    'Bouvet Island': 'Bouvet Island',
    'Brazil': 'Brazil',
    'British Indian Ocean Territory': 'British Indian Ocean Territory',
    'British Virgin Islands': 'Virgin Islands, British',
    'Brunei Darussalam': 'Brunei Darussalam',
    'Bulgaria': 'Bulgaria',
    'Burkina Faso': 'Burkina Faso',
    'Burundi': 'Burundi',
    'Cabo Verde': 'Cabo Verde',
    'Cambodia': 'Cambodia',
    'Cameroon': 'Cameroon',
    'Canada': 'Canada',
    'Cayman Islands': 'Cayman Islands',
    'Central African Republic': 'Central African Republic',
    'Chad': 'Chad',
    'Chile': 'Chile',
    'China': 'China',
    'Christmas Island': 'Christmas Island',
    'Cocos (Keeling) Islands': 'Cocos (Keeling) Islands',
    'Colombia': 'Colombia',
    'Commonwealth of Dominica': 'Dominica',
    'Commonwealth of the Bahamas': 'Bahamas',
    'Commonwealth of the Northern Mariana Islands': 'Northern Mariana Islands',
    'Comoros': 'Comoros',
    'Congo': 'Congo',
    'Congo, The Democratic Republic of the': 'Congo, The Democratic Republic of the',
    'Cook Islands': 'Cook Islands',
    'Costa Rica': 'Costa Rica',
    'Croatia': 'Croatia',
    'Cuba': 'Cuba',
    'Curaçao': 'Curaçao',
    'Cyprus': 'Cyprus',
    'Czech Republic': 'Czechia',
    'Czechia': 'Czechia',
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Democratic People's Republic of Korea": "Korea, Democratic People's Republic of",
    'Democratic Republic of Sao Tome and Principe': 'Sao Tome and Principe',
    'Democratic Republic of Timor-Leste': 'Timor-Leste',
    'Democratic Socialist Republic of Sri Lanka': 'Sri Lanka',
    'Denmark': 'Denmark',
    'Djibouti': 'Djibouti',
    'Dominica': 'Dominica',
    'Dominican Republic': 'Dominican Republic',
    'Eastern Republic of Uruguay': 'Uruguay',
    'Ecuador': 'Ecuador',
    'Egypt': 'Egypt',
    'El Salvador': 'El Salvador',
    'Equatorial Guinea': 'Equatorial Guinea',
    'Eritrea': 'Eritrea',
    'Estonia': 'Estonia',
    'Eswatini': 'Eswatini',
    'Ethiopia': 'Ethiopia',
    'Falkland Islands (Malvinas)': 'Falkland Islands (Malvinas)',
    'Faroe Islands': 'Faroe Islands',
    'Federal Democratic Republic of Ethiopia': 'Ethiopia',
    'Federal Democratic Republic of Nepal': 'Nepal',
    'Federal Republic of Germany': 'Germany',
    'Federal Republic of Nigeria': 'Nigeria',
    'Federal Republic of Somalia': 'Somalia',
    'Federated States of Micronesia': 'Micronesia, Federated States of',
    'Federative Republic of Brazil': 'Brazil',
    'Fiji': 'Fiji',
    'Finland': 'Finland',
    'France': 'France',
    'French Guiana': 'French Guiana',
    'French Polynesia': 'French Polynesia',
    'French Republic': 'France',
    'French Southern Territories': 'French Southern Territories',
    'Gabon': 'Gabon',
    'Gabonese Republic': 'Gabon',
    'Gambia': 'Gambia',
    'Georgia': 'Georgia',
    'Germany': 'Germany',
    'Ghana': 'Ghana',
    'Gibraltar': 'Gibraltar',
    'Grand Duchy of Luxembourg': 'Luxembourg',
    'Greece': 'Greece',
    'Greenland': 'Greenland',
    'Grenada': 'Grenada',
    'Guadeloupe': 'Guadeloupe',
    'Guam': 'Guam',
    'Guatemala': 'Guatemala',
    'Guernsey': 'Guernsey',
    'Guinea': 'Guinea',
    'Guinea-Bissau': 'Guinea-Bissau',
    'Guyana': 'Guyana',
    'Haiti': 'Haiti',
    'Hashemite Kingdom of Jordan': 'Jordan',
    'Heard Island and McDonald Islands': 'Heard Island and McDonald Islands',
    'Hellenic Republic': 'Greece',
    'Holy See (Vatican City State)': 'Holy See (Vatican City State)',
    'Honduras': 'Honduras',
    'Hong Kong': 'Hong Kong',
    'Hong Kong Special Administrative Region of China': 'Hong Kong',
    'Hungary': 'Hungary',
    'Iceland': 'Iceland',
    'Independent State of Papua New Guinea': 'Papua New Guinea',
    'Independent State of Samoa': 'Samoa',
    'India': 'India',
    'Indonesia': 'Indonesia',
    'Iran': 'Iran, Islamic Republic of',
    'Iran, Islamic Republic of': 'Iran, Islamic Republic of',
    'Iraq': 'Iraq',
    'Ireland': 'Ireland',
    'Islamic Republic of Afghanistan': 'Afghanistan',
    'Islamic Republic of Iran': 'Iran, Islamic Republic of',
    'Islamic Republic of Mauritania': 'Mauritania',
    'Islamic Republic of Pakistan': 'Pakistan',
    'Isle of Man': 'Isle of Man',
    'Israel': 'Israel',
    'Italian Republic': 'Italy',
    'Italy': 'Italy',
    'Jamaica': 'Jamaica',
    'Japan': 'Japan',
    'Jersey': 'Jersey',
    'Jordan': 'Jordan',
    'Kazakhstan': 'Kazakhstan',
    'Kenya': 'Kenya',
    'Kingdom of Bahrain': 'Bahrain',
    'Kingdom of Belgium': 'Belgium',
    'Kingdom of Bhutan': 'Bhutan',
    'Kingdom of Cambodia': 'Cambodia',
    'Kingdom of Denmark': 'Denmark',
    'Kingdom of Eswatini': 'Eswatini',
    'Kingdom of Lesotho': 'Lesotho',
    'Kingdom of Morocco': 'Morocco',
    'Kingdom of Norway': 'Norway',
    'Kingdom of Saudi Arabia': 'Saudi Arabia',
    'Kingdom of Spain': 'Spain',
    'Kingdom of Sweden': 'Sweden',
    'Kingdom of Thailand': 'Thailand',
    'Kingdom of the Netherlands': 'Netherlands',
    'Kingdom of Tonga': 'Tonga',
    'Kiribati': 'Kiribati',
    "Korea, Democratic People's Republic of": "Korea, Democratic People's Republic of",
    'Korea, Republic of': 'Korea, Republic of',
    'Kuwait': 'Kuwait',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Kyrgyzstan': 'Kyrgyzstan',
    "Lao People's Democratic Republic": "Lao People's Democratic Republic",
    'Laos': "Lao People's Democratic Republic",
    'Latvia': 'Latvia',
    'Lebanese Republic': 'Lebanon',
    'Lebanon': 'Lebanon',
    'Lesotho': 'Lesotho',
    'Liberia': 'Liberia',
    'Libya': 'Libya',
    'Liechtenstein': 'Liechtenstein',
    'Lithuania': 'Lithuania',
    'Luxembourg': 'Luxembourg',
    'Macao': 'Macao',
    'Macao Special Administrative Region of China': 'Macao',
    'Madagascar': 'Madagascar',
    'Malawi': 'Malawi',
    'Malaysia': 'Malaysia',
    'Maldives': 'Maldives',
    'Mali': 'Mali',
    'Malta': 'Malta',
    'Marshall Islands': 'Marshall Islands',
    'Martinique': 'Martinique',
    'Mauritania': 'Mauritania',
    'Mauritius': 'Mauritius',
    'Mayotte': 'Mayotte',
    'Mexico': 'Mexico',
    'Micronesia, Federated States of': 'Micronesia, Federated States of',
    'Moldova': 'Moldova, Republic of',
    'Moldova, Republic of': 'Moldova, Republic of',
    'Monaco': 'Monaco',
    'Mongolia': 'Mongolia',
    'Montenegro': 'Montenegro',
    'Montserrat': 'Montserrat',
    'Morocco': 'Morocco',
    'Mozambique': 'Mozambique',
    'Myanmar': 'Myanmar',
    'Namibia': 'Namibia',
    'Nauru': 'Nauru',
    'Nepal': 'Nepal',
    'Netherlands': 'Netherlands',
    'New Caledonia': 'New Caledonia',
    'New Zealand': 'New Zealand',
    'Nicaragua': 'Nicaragua',
    'Niger': 'Niger',
    'Nigeria': 'Nigeria',
    'Niue': 'Niue',
    'Norfolk Island': 'Norfolk Island',
    'North Korea': "Korea, Democratic People's Republic of",
    'North Macedonia': 'North Macedonia',
    'Northern Mariana Islands': 'Northern Mariana Islands',
    'Norway': 'Norway',
    'Oman': 'Oman',
    'Pakistan': 'Pakistan',
    'Palau': 'Palau',
    'Palestine, State of': 'Palestine, State of',
    'Panama': 'Panama',
    'Papua New Guinea': 'Papua New Guinea',
    'Paraguay': 'Paraguay',
    "People's Democratic Republic of Algeria": 'Algeria',
    "People's Republic of Bangladesh": 'Bangladesh',
    "People's Republic of China": 'China',
    'Peru': 'Peru',
    'Philippines': 'Philippines',
    'Pitcairn': 'Pitcairn',
    'Plurinational State of Bolivia': 'Bolivia, Plurinational State of',
    'Poland': 'Poland',
    'Portugal': 'Portugal',
    'Portuguese Republic': 'Portugal',
    'Principality of Andorra': 'Andorra',
    'Principality of Liechtenstein': 'Liechtenstein',
    'Principality of Monaco': 'Monaco',
    'Puerto Rico': 'Puerto Rico',
    'Qatar': 'Qatar',
    'Republic of Albania': 'Albania',
    'Republic of Angola': 'Angola',
    'Republic of Armenia': 'Armenia',
    'Republic of Austria': 'Austria',
    'Republic of Azerbaijan': 'Azerbaijan',
    'Republic of Belarus': 'Belarus',
    'Republic of Benin': 'Benin',
    'Republic of Bosnia and Herzegovina': 'Bosnia and Herzegovina',
    'Republic of Botswana': 'Botswana',
    'Republic of Bulgaria': 'Bulgaria',
    'Republic of Burundi': 'Burundi',
    'Republic of Cabo Verde': 'Cabo Verde',
    'Republic of Cameroon': 'Cameroon',
    'Republic of Chad': 'Chad',
    'Republic of Chile': 'Chile',
    'Republic of Colombia': 'Colombia',
    'Republic of Costa Rica': 'Costa Rica',
    'Republic of Croatia': 'Croatia',
    'Republic of Cuba': 'Cuba',
    'Republic of Cyprus': 'Cyprus',
    "Republic of Côte d'Ivoire": "Côte d'Ivoire",
    'Republic of Djibouti': 'Djibouti',
    'Republic of Ecuador': 'Ecuador',
    'Republic of El Salvador': 'El Salvador',
    'Republic of Equatorial Guinea': 'Equatorial Guinea',
    'Republic of Estonia': 'Estonia',
    'Republic of Fiji': 'Fiji',
    'Republic of Finland': 'Finland',
    'Republic of Ghana': 'Ghana',
    'Republic of Guatemala': 'Guatemala',
    'Republic of Guinea': 'Guinea',
    'Republic of Guinea-Bissau': 'Guinea-Bissau',
    'Republic of Guyana': 'Guyana',
    'Republic of Haiti': 'Haiti',
    'Republic of Honduras': 'Honduras',
    'Republic of Iceland': 'Iceland',
    'Republic of India': 'India',
    'Republic of Indonesia': 'Indonesia',
    'Republic of Iraq': 'Iraq',
    'Republic of Kazakhstan': 'Kazakhstan',
    'Republic of Kenya': 'Kenya',
    'Republic of Kiribati': 'Kiribati',
    'Republic of Latvia': 'Latvia',
    'Republic of Liberia': 'Liberia',
    'Republic of Lithuania': 'Lithuania',
    'Republic of Madagascar': 'Madagascar',
    'Republic of Malawi': 'Malawi',
    'Republic of Maldives': 'Maldives',
    'Republic of Mali': 'Mali',
    'Republic of Malta': 'Malta',
    'Republic of Mauritius': 'Mauritius',
    'Republic of Moldova': 'Moldova, Republic of',
    'Republic of Mozambique': 'Mozambique',
    'Republic of Myanmar': 'Myanmar',
    'Republic of Namibia': 'Namibia',
    'Republic of Nauru': 'Nauru',
    'Republic of Nicaragua': 'Nicaragua',
    'Republic of North Macedonia': 'North Macedonia',
    'Republic of Palau': 'Palau',
    'Republic of Panama': 'Panama',
    'Republic of Paraguay': 'Paraguay',
    'Republic of Peru': 'Peru',
    'Republic of Poland': 'Poland',
    'Republic of San Marino': 'San Marino',
    'Republic of Senegal': 'Senegal',
    'Republic of Serbia': 'Serbia',
    'Republic of Seychelles': 'Seychelles',
    'Republic of Sierra Leone': 'Sierra Leone',
    'Republic of Singapore': 'Singapore',
    'Republic of Slovenia': 'Slovenia',
    'Republic of South Africa': 'South Africa',
    'Republic of South Sudan': 'South Sudan',
    'Republic of Suriname': 'Suriname',
    'Republic of Tajikistan': 'Tajikistan',
    'Republic of the Congo': 'Congo',
    'Republic of the Gambia': 'Gambia',
    'Republic of the Marshall Islands': 'Marshall Islands',
    'Republic of the Niger': 'Niger',
    'Republic of the Philippines': 'Philippines',
    'Republic of the Sudan': 'Sudan',
    'Republic of Trinidad and Tobago': 'Trinidad and Tobago',
    'Republic of Tunisia': 'Tunisia',
    'Republic of Türkiye': 'Türkiye',
    'Republic of Uganda': 'Uganda',
    'Republic of Uzbekistan': 'Uzbekistan',
    'Republic of Vanuatu': 'Vanuatu',
    'Republic of Yemen': 'Yemen',
    'Republic of Zambia': 'Zambia',
    'Republic of Zimbabwe': 'Zimbabwe',
    'Romania': 'Romania',
    'Russian Federation': 'Russian Federation',
    'Rwanda': 'Rwanda',
    'Rwandese Republic': 'Rwanda',
    'Réunion': 'Réunion',
    'Saint Barthélemy': 'Saint Barthélemy',
    'Saint Helena, Ascension and Tristan da Cunha': 'Saint Helena, Ascension and Tristan da Cunha',
    'Saint Kitts and Nevis': 'Saint Kitts and Nevis',
    'Saint Lucia': 'Saint Lucia',
    'Saint Martin (French part)': 'Saint Martin (French part)',
    'Saint Pierre and Miquelon': 'Saint Pierre and Miquelon',
    'Saint Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Samoa': 'Samoa',
    'San Marino': 'San Marino',
    'Sao Tome and Principe': 'Sao Tome and Principe',
    'Saudi Arabia': 'Saudi Arabia',
    'Senegal': 'Senegal',
    'Serbia': 'Serbia',
    'Seychelles': 'Seychelles',
    'Sierra Leone': 'Sierra Leone',
    'Singapore': 'Singapore',
    'Sint Maarten (Dutch part)': 'Sint Maarten (Dutch part)',
    'Slovak Republic': 'Slovakia',
    'Slovakia': 'Slovakia',
    'Slovenia': 'Slovenia',
    'Socialist Republic of Viet Nam': 'Viet Nam',
    'Solomon Islands': 'Solomon Islands',
    'Somalia': 'Somalia',
    'South Africa': 'South Africa',
    'South Georgia and the South Sandwich Islands': 'South Georgia and the South Sandwich Islands',
    'South Korea': 'Korea, Republic of',
    'South Sudan': 'South Sudan',
    'Spain': 'Spain',
    'Sri Lanka': 'Sri Lanka',
    'State of Israel': 'Israel',
    'State of Kuwait': 'Kuwait',
    'State of Qatar': 'Qatar',
    'Sudan': 'Sudan',
    'Sultanate of Oman': 'Oman',
    'Suriname': 'Suriname',
    'Svalbard and Jan Mayen': 'Svalbard and Jan Mayen',
    'Sweden': 'Sweden',
    'Swiss Confederation': 'Switzerland',
    'Switzerland': 'Switzerland',
    'Syria': 'Syrian Arab Republic',
    'Syrian Arab Republic': 'Syrian Arab Republic',
    'Taiwan': 'Taiwan, Province of China',
    'Taiwan, Province of China': 'Taiwan, Province of China',
    'Tajikistan': 'Tajikistan',
    'Tanzania': 'Tanzania, United Republic of',
    'Tanzania, United Republic of': 'Tanzania, United Republic of',
    'Thailand': 'Thailand',
    'the State of Eritrea': 'Eritrea',
    'the State of Palestine': 'Palestine, State of',
    'Timor-Leste': 'Timor-Leste',
    'Togo': 'Togo',
    'Togolese Republic': 'Togo',
    'Tokelau': 'Tokelau',
    'Tonga': 'Tonga',
    'Trinidad and Tobago': 'Trinidad and Tobago',
    'Tunisia': 'Tunisia',
    'Turkmenistan': 'Turkmenistan',
    'Turks and Caicos Islands': 'Turks and Caicos Islands',
    'Tuvalu': 'Tuvalu',
    'Türkiye': 'Türkiye',
    'Uganda': 'Uganda',
    'Ukraine': 'Ukraine',
    'Union of the Comoros': 'Comoros',
    'United Arab Emirates': 'United Arab Emirates',
    'United Kingdom': 'United Kingdom',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'United Mexican States': 'Mexico',
    'United Republic of Tanzania': 'Tanzania, United Republic of',
    'United States': 'United States',
    'United States Minor Outlying Islands': 'United States Minor Outlying Islands',
    'United States of America': 'United States',
    'Uruguay': 'Uruguay',
    'Uzbekistan': 'Uzbekistan',
    'Vanuatu': 'Vanuatu',
    'Venezuela': 'Venezuela, Bolivarian Republic of',
    'Venezuela, Bolivarian Republic of': 'Venezuela, Bolivarian Republic of',
    'Viet Nam': 'Viet Nam',
    'Vietnam': 'Viet Nam',
    'Virgin Islands of the United States': 'Virgin Islands, U.S.',
    'Virgin Islands, British': 'Virgin Islands, British',
    'Virgin Islands, U.S.': 'Virgin Islands, U.S.',
    'Wallis and Futuna': 'Wallis and Futuna',
    'Western Sahara': 'Western Sahara',
    'Yemen': 'Yemen',
    'Zambia': 'Zambia',
    'Zimbabwe': 'Zimbabwe',
    'Åland Islands': 'Åland Islands',
}

# Common colloquial names and historical spellings not always present in ISO.
ALIASES = {
    "usa": "United States", "u s a": "United States",
    "us": "United States", "u s": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom", "u k": "United Kingdom",
    "great britain": "United Kingdom", "britain": "United Kingdom",
    "england": "United Kingdom", "scotland": "United Kingdom",
    "wales": "United Kingdom", "northern ireland": "United Kingdom",
    "russia": "Russian Federation", "south korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "taiwan": "Taiwan, Province of China",
    "vietnam": "Viet Nam", "bolivia": "Bolivia, Plurinational State of",
    "venezuela": "Venezuela, Bolivarian Republic of",
    "tanzania": "Tanzania, United Republic of", "moldova": "Moldova, Republic of",
    "iran": "Iran, Islamic Republic of", "syria": "Syrian Arab Republic",
    "laos": "Lao People's Democratic Republic", "brunei": "Brunei Darussalam",
    "czech republic": "Czechia", "ivory coast": "Côte d'Ivoire",
    "curacao": "Curaçao", "hong kong": "Hong Kong",
    "macau": "Macao", "palestine": "Palestine, State of",
    "cape verde": "Cabo Verde", "swaziland": "Eswatini",
    "republic of the congo": "Congo", "democratic republic of the congo": "Congo, The Democratic Republic of the",
    "dr congo": "Congo, The Democratic Republic of the",
    "south africa": "South Africa", "the netherlands": "Netherlands",
    "the bahamas": "Bahamas", "the gambia": "Gambia",
    # Explicit historical and informal endings observed in the location audit.
    # These are exact suffix aliases, not fuzzy city-based geolocation.
    "turkey": "Türkiye", "turkiye": "Türkiye",
    "palestinian territories": "Palestine, State of",
    "palestinian territory": "Palestine, State of",
    "libyan arab jamahiriya": "Libya",
    "macedonia": "North Macedonia",
    "falkland islands": "Falkland Islands (Malvinas)",
    "micronesia": "Micronesia, Federated States of",
}

# These are present in ISO, but a terminal string alone is insufficiently
# distinct from a sub-national place or among different countries.
AMBIGUOUS_TERMS = frozenset({"georgia", "jersey", "congo", "korea", "virgin islands"})
US_TERRITORIES = frozenset({
    "puerto rico", "guam", "american samoa", "northern mariana islands",
    "united states minor outlying islands", "us virgin islands",
    "u s virgin islands", "united states virgin islands",
    "virgin islands u s", "virgin islands of the united states",
})

# Audit evidence only: these names never expand the frozen state-code cohort.
US_STATE_NAMES = frozenset("""alabama|alaska|arizona|arkansas|california|colorado|
connecticut|delaware|florida|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|
louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|
montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|
north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|
south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|
west virginia|wisconsin|wyoming|district of columbia""".replace("\n", "").split("|"))
# Georgia deliberately remains ambiguous.


def normalize(text):
    """Fold accents/punctuation for comparing names; never guess fuzzy names."""
    if text is None or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


COUNTRY_LOOKUP = {normalize(k): v for k, v in COUNTRY_NAMES.items()}
COUNTRY_LOOKUP.update({normalize(k): v for k, v in ALIASES.items()})


def classify_location(location):
    """Return (scope, country_or_region, evidence) for one location string.

    Scope is 'us_candidate', 'explicit_us', 'us_territory',
    'explicit_non_us', or 'unknown'. Only explicit_non_us is eligible for
    the international cohort. US state abbreviations are not country codes.
    """
    if location is None or pd.isna(location) or not str(location).strip():
        return "unknown", None, "missing_location"
    raw = str(location).strip()
    last = raw.rsplit(",", 1)[-1].strip()
    normalized = normalize(last)

    # Respect the already-established U.S. state-code cohort definition.
    if "," in raw and re.fullmatch(r"[A-Za-z]{2}", last) and last.upper() in US_STATE_CODES:
        return "us_candidate", "United States (state proxy)", "us_state_suffix"
    # Longest complete suffix first: official names may themselves contain
    # commas, e.g. Virgin Islands, U.S. Never fall back to just U.S. there.
    parts = raw.split(",")
    country = None
    for i in range(len(parts)):
        suffix = normalize(",".join(parts[i:]))
        candidate = COUNTRY_LOOKUP.get(suffix)
        if suffix in US_TERRITORIES or normalize(candidate) in US_TERRITORIES:
            return "us_territory", None, "us_territory_excluded"
        if suffix not in AMBIGUOUS_TERMS and candidate is not None:
            country = candidate
            break
    if normalized in US_STATE_NAMES and country is None:
        return "explicit_us", "United States", "full_state_name_outside_proxy"
    if normalized in AMBIGUOUS_TERMS:
        if country is None:
            return "unknown", None, "ambiguous_country_name"
    if country is None:
        return "unknown", None, "no_explicit_country_name"
    if normalize(country) in {"united states", "united states of america"}:
        return "explicit_us", "United States", "explicit_us_name"
    # Country names are not fuzzy-inferred; province-only/ISO2 endings are
    # intentionally left as unknown, e.g. Toronto, ON or Berlin, BE.
    return "explicit_non_us", country, "explicit_country_or_region_name"


def classify_locations(locations):
    """Return indexed classification columns without looking at outcomes."""
    if not isinstance(locations, pd.Series):
        locations = pd.Series(locations)
    records = [classify_location(loc) for loc in locations]
    return pd.DataFrame(records, columns=["scope", "country_region", "evidence"], index=locations.index)


def international_mask(df):
    if "location" not in df.columns:
        raise KeyError("Missing location column")
    return classify_locations(df["location"])["scope"].eq("explicit_non_us")


def location_scope_audit(df):
    if "location" not in df:
        raise KeyError("Missing location column")
    classes = classify_locations(df["location"])
    overview = classes.groupby(["scope", "evidence"], dropna=False).size().rename("n").reset_index()
    counts = (classes.loc[classes["scope"].eq("explicit_non_us"), "country_region"]
              .value_counts().rename_axis("country_region").reset_index(name="n"))
    uncertain = (df.loc[classes["scope"].eq("unknown"), ["location"]]
                 .assign(evidence=classes.loc[classes["scope"].eq("unknown"), "evidence"].to_numpy())
                 .groupby(["location", "evidence"], dropna=False)
                 .size().rename("n").reset_index().sort_values("n", ascending=False))
    proxy = us_state_proxy_mask(df)
    if not classes["scope"].eq("us_candidate").equals(proxy):
        raise AssertionError("U.S. proxy and geographic classifier disagree")
    if (classes["scope"].eq("explicit_non_us") & proxy).any():
        raise AssertionError("U.S. and international proxy memberships overlap")
    return classes, overview, counts, uncertain


def align_international_to_global_split(clean, global_train_ids, global_test_ids):
    """Preserve original split membership; validate scope and all labels."""
    if not international_mask(clean).all():
        raise ValueError("International cohort includes uncertain/U.S. locations")
    # Generic split invariants are already covered by the team's U.S. helper.
    return align_to_global_split(clean, global_train_ids, global_test_ids)
