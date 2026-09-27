"""
District Mapping for West Bengal Medical Colleges and Healthcare Institutions.
Maps all 64 unique institutes to their official West Bengal district.
"""

INSTITUTE_TO_DISTRICT: dict[str, str] = {
    # Kolkata
    "ALL INDIA INSTITUTE OF HYGIENE AND PUBLIC HEALTH": "Kolkata",
    "CALCUTTA NATIONAL MEDICAL COLLEGE KOLKATA": "Kolkata",
    "CHITTARANJAN SEVA SADAN COLLEGE OF OBS. GYNAE AND CHILD HEALTH,KOLKATA": "Kolkata",
    "COMMAND HOSPITAL EC, ALIPORE, KOLKATA": "Kolkata",
    "DR. B C ROY POST GRADUATE INSTITUTE OF PAEDIATRIC SCIENCES": "Kolkata",
    "ESI PGIMSR AND ESIC MEDICAL COLLEGE JOKA KOLKATA": "Kolkata",
    "INFECTIOUS DISEASES AND BELIAGHATA GENERAL HOSPITAL": "Kolkata",
    "INSTITUTE OF CHILD HEALTH": "Kolkata",
    "INSTITUTE OF POST GRADUATE MEDICAL EDUCATION AND RESEARCH KOLKATA": "Kolkata",
    "KPC MEDICAL COLLEGE": "Kolkata",
    "M R BANGUR HOSPITAL": "Kolkata",
    "MEDICAL COLLEGE KOLKATA": "Kolkata",
    "NIL RATAN SIRCAR MEDICAL COLLEGE": "Kolkata",
    "R G KAR MEDICAL COLLEGE": "Kolkata",
    "RAMAKRISHNA MISSION SEVA PRATISHTHAN VIVEKANANDA INSTITUTE OF MEDICAL SCIENCES, KOLKATA": "Kolkata",
    "RAMKRISHNA MISSION SEVA PRASITHAN VIVEKANANDA INSTITUTE OF MEDICAL SCIENCE": "Kolkata",
    "SCHOOL OF TROPICAL MEDICINE, KOLKATA": "Kolkata",
    "VIDYASAGAR STATE GENERAL HOSPITAL": "Kolkata",

    # North 24 Parganas
    "BARASAT GOVERNMENT MEDICAL COLLEGE AND HOSPITAL": "North 24 Parganas",
    "COLLEGE OF MEDICINE AND SAGORE DUTTA HOSPITAL": "North 24 Parganas",
    "DR.B.N.BOSE SUB DIVISIONAL HOSPITAL": "North 24 Parganas",
    "SREE BALARAM SEVA MANDIR S.G. HOSPITAL": "North 24 Parganas",

    # South 24 Parganas
    "DIAMOND HARBOUR GOVERNMENT MEDICAL COLLEGE AND HOSPITAL": "South 24 Parganas",
    "JAGANNATH GUPTA INSTITUTE OF MEDICAL SCIENCES AND HOSPITAL": "South 24 Parganas",

    # Howrah
    "DISTRICT HOSPITAL HOWRAH": "Howrah",
    "JIS SCHOOL OF MEDICAL SCIENCE & RESEARCH": "Howrah",

    # Hooghly
    "CHANDANNAGAR SUB DIVISION HOSPITAL": "Hooghly",
    "IMAMBARA DISTRICT HOSPITAL": "Hooghly",
    "PRAFULLA CHANDRA SEN GOVT. MEDICAL COLLEGE AND HOSPITAL": "Hooghly",
    "PRAFULLA CHANDRA SEN GOVT.MEDICAL COLLEGE AND HOSPITAL": "Hooghly",
    "WALSH S.D.H., SERAMPORE": "Hooghly",

    # Nadia
    "COLLEGE OF MEDICINE AND JNM HOSPITAL KALYANI": "Nadia",
    "DISTRICT HOSPITAL NADIA": "Nadia",
    "RANAGHAT SUB DIVISIONAL HOSPITAL": "Nadia",

    # Purba Bardhaman
    "BURDWAN MEDICAL COLLEGE": "Purba Bardhaman",
    "KALNA SD AND SS HOSPITAL": "Purba Bardhaman",

    # Paschim Bardhaman
    "ASANSOL DISTRICT HOSPITAL": "Paschim Bardhaman",
    "DURGAPUR SUB DIVISIONAL HOSPITAL": "Paschim Bardhaman",
    "GOURI DEVI INSTITUTE OF MEDICAL SCIENCES AND HOSPITAL": "Paschim Bardhaman",
    "IQ CITY MEDICAL COLLEGE, DURGAPUR": "Paschim Bardhaman",
    "SHRI RAMKRISHNA INSTITUTE OF MEDICAL SCIENCES & SANAKA HOSPITALS": "Paschim Bardhaman",

    # Birbhum
    "BOLPUR SUB DIVISION HOSPITAL": "Birbhum",
    "RAMPURHAT GOVT. MEDICAL COLLEGE AND HOSPITAL": "Birbhum",
    "SURI SADAR HOSPITAL": "Birbhum",

    # Bankura
    "BANKURA SAMMILANI MEDICAL COLLEGE": "Bankura",

    # Purulia
    "DEBEN MAHATO SADAR HOSPITAL, PURULIA": "Purulia",

    # Paschim Medinipur
    "KHARAGPUR SUB DIVISIONAL HOSPITAL": "Paschim Medinipur",
    "MIDNAPORE MEDICAL COLLEGE": "Paschim Medinipur",

    # Purba Medinipur
    "ICARE INSTITUTE OF MEDICAL SCIENCES AND RESEARCH HALDIA": "Purba Medinipur",
    "TAMRALIPTO GOVERNMENT MEDICAL COLLEGE": "Purba Medinipur",

    # Murshidabad
    "DOMKAL SUB DIVISIONAL AND SUPER SPECIALITY HOSPITAL": "Murshidabad",
    "JANGIPUR S. D. HOSPITAL": "Murshidabad",
    "LALBAGH SUB DIVISION HOSPITAL": "Murshidabad",
    "MURSHIDABAD MEDICAL COLLEGE AND HOSPITAL": "Murshidabad",

    # Malda
    "MALDA MEDICAL COLLEGE AND HOSPITAL": "Malda",

    # Uttar Dinajpur
    "ISLAMPUR SUB DIVISIONAL HOSPITAL": "Uttar Dinajpur",
    "RAIGANJ GOVERNMENT MEDICAL COLLEGE AND HOSPITAL": "Uttar Dinajpur",

    # Dakshin Dinajpur
    "BALURGHAT DH AND SSH": "Dakshin Dinajpur",

    # Darjeeling
    "DISTRICT HOSPITAL DARJEELING": "Darjeeling",
    "NORTH BENGAL MEDICAL COLLEGE AND HOSPITAL": "Darjeeling",
    "SILIGURI DISTRICT HOSPITAL": "Darjeeling",

    # Jalpaiguri
    "Jalpaiguri Government Medical College and Hospital (Formerly DISTRICT HOSPITAL) JALPAIGURI": "Jalpaiguri",

    # Alipurduar
    "DISTRICT HOSPITAL ALIPURDUAR": "Alipurduar",

    # Cooch Behar
    "MAHARAJA JITENDRA NARAYAN MEDICAL COLLEGE AND HOSPITAL": "Cooch Behar",
}

ALL_DISTRICTS: list[str] = sorted(list(set(INSTITUTE_TO_DISTRICT.values())))


def get_district(institute: str) -> str:
    """Returns the district for a given medical college or hospital."""
    clean_inst = institute.strip()
    return INSTITUTE_TO_DISTRICT.get(clean_inst, "West Bengal")
