"""Name-based description of the BRFSS 2015 columns used in project 1.

Everything here is keyed by the variable *name* from the CSV header (see
``preprocessing.load_feature_names``), never by column index, so the tables stay
correct even if columns are added, removed or reordered.

Sources: CDC BRFSS 2015 codebook, variable layout and "Calculated Variables in
the 2015 Data File" (https://www.cdc.gov/brfss/annual_data/annual_2015.html).

Conventions used by the BRFSS questionnaire:
* 1-digit answers: 7 = "don't know / not sure", 9 = "refused".
* 2-digit answers: 77 / 99 (sometimes 98) = unknown, 88 = "none" (i.e. zero).
* 3-digit answers: 777 / 999 = unknown, 888 = "none", 555 = "never".
* Yes/No questions: 1 = yes, 2 = no.
* Calculated variables (prefixed with "_" or ending with "_") usually use
  9 (or 900 / 99900 / 99999 for numeric ones) for "missing".
"""

# --------------------------------------------------------------------------- #
# Columns to remove before modelling
# --------------------------------------------------------------------------- #

# Interview bookkeeping, phone sampling questions, survey weights and IDs.
# They describe how the person was interviewed, not their health.
ADMIN_COLUMNS = (
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2",
    "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT", "QSTVER", "QSTLANG",
    "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CHISPNC",
    "_CRACE1", "_CPRACE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
)

# Dates (MMYYYY), free-text activity codes and nominal codes with dozens of
# levels (place of flu shot, caregiver relationship, ...). Not worth encoding.
UNUSABLE_COLUMNS = (
    "FLSHTMY2", "HIVTSTD3", "PCDMDECN", "EXACTOT1", "EXACTOT2", "EXRACT11",
    "EXRACT21", "IMFVPLAC", "WHRTST10", "CRGVREL1", "CRGVPRB1", "RCSRLTN2",
    "LONGWTCH",
)

# Raw answers for which the CDC already provides a cleaned calculated version.
# Raw weight/height mix pounds/kg and feet/metres in one field, raw food and
# exercise frequencies mix "per day/week/month" codes, so the calculated
# versions (WTKG3, HTM4, _BMI5, DROCDY3_, FRUTDA1_, PAFREQ1_, ...) are better.
# Age and race appear in several groupings; we keep _AGE80 and _RACE.
REDUNDANT_COLUMNS = (
    "WEIGHT2", "HEIGHT3", "HTIN4",
    "ALCDAY5",
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",
    "EXEROFT1", "EXERHMM1", "EXEROFT2", "EXERHMM2", "STRENGTH",
    "_AGEG5YR", "_AGE65YR", "_AGE_G",
    "_RACEG21", "_RACEGR3", "_RACE_G1", "_PRACE1", "_MRACE1", "_HISPANC",
    "_RFBMI5", "_CHLDCNT", "_EDUCAG", "_INCOMG", "_RFSMOK3",
    "_MISFRTN", "_MISVEGN", "_FRTRESP", "_VEGRESP",
    "_FRTLT1", "_VEGLT1", "_FRT16", "_VEG23",
)

# State FIPS code: nominal with ~53 levels. Dropped by default, can be one-hot
# encoded instead with ``keep_state=True`` in the pipeline.
STATE_COLUMN = "_STATE"

# --------------------------------------------------------------------------- #
# Special answer codes
# --------------------------------------------------------------------------- #

# 1-digit questions where 7 = don't know and 9 = refused.
_CODES_7_9 = (
    "GENHLTH", "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4",
    "BPMEDS", "BLOODCHO", "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3",
    "ASTHNOW", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2",
    "CHCKIDNY", "DIABETE3", "RENTHOM1", "NUMHHOL2", "NUMPHON2", "CPDEMO1",
    "VETERAN3", "INTERNET", "PREGNANT", "QLACTLM2", "USEEQUIP", "BLIND",
    "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2",
    "STOPSMK2", "USENOW3", "EXERANY2", "LMTJOIN3", "ARTHDIS2", "ARTHSOCL",
    "SEATBELT", "FLUSHOT6", "PNEUVAC3", "HIVTST6", "PDIABTST", "PREDIAB1",
    "INSULIN", "EYEEXAM", "DIABEYE", "DIABEDU", "CAREGIV1", "CRGVLNG1",
    "CRGVHRS1", "CRGVPERS", "CRGVHOUS", "CRGVMST2", "CRGVEXPT", "VIDFCLT2",
    "VIREDIF3", "VIPRFVS2", "VIEYEXM2", "VIINSUR2", "VICTRCT4", "VIGLUMA2",
    "VIMACDG2", "CIMEMLOS", "CDHOUSE", "CDASSIST", "CDHELP", "CDSOCIAL",
    "CDDISCUS", "WTCHSALT", "DRADVISE", "ASATTACK", "ASYMPTOM", "ASNOSLEP",
    "ASTHMED3", "ASINHALR", "HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF",
    "RLIVPAIN", "RDUCHART", "RDUCSTRK", "ARTTODAY", "ARTHWGT", "ARTHEXER",
    "ARTHEDU", "TETANUS", "HPVADVC2", "SHINGLE2", "HADMAM", "HOWLONG",
    "HADPAP2", "LASTPAP2", "HPVTEST", "HPLSTTST", "HADHYST2", "PROFEXAM",
    "LENGEXAM", "BLDSTOOL", "LSTBLDS3", "HADSIGM3", "HADSGCO1", "LASTSIG3",
    "PCPSAAD2", "PCPSADI1", "PCPSARE1", "PSATEST1", "PSATIME", "PCPSARS1",
    "PCPSADE1", "SCNTMNY1", "SCNTMEL1", "SCNTPAID", "SCNTLPAD", "SXORIENT",
    "TRNSGNDR", "CASTHDX2", "CASTHNO2", "EMTSUPRT", "LSATISFY", "MISTMNT",
    "ADANXEV", "DRNKANY5",
)

# 1-digit questions where 7 is a *valid* answer, only 9 = refused.
# (e.g. EMPLOY1: 7 = retired; MARITAL/EDUCA have no "don't know" option.)
_CODES_9 = ("SEX", "MARITAL", "EDUCA", "EMPLOY1", "RCSGENDR")

# 2-digit questions: 77 = don't know, 98 = don't know (some items), 99 = refused.
_CODES_77_98_99 = (
    "PHYSHLTH", "MENTHLTH", "POORHLTH", "DIABAGE2", "CHILDREN", "INCOME2",
    "LASTSMK2", "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "JOINPAIN", "DOCTDIAB",
    "CHKHEMO3", "FEETCHK", "PAINACT2", "QLMENTL2", "QLSTRES2", "QLHLTH2",
    "VINOCRE2", "ASTHMAGE", "ASERVIST", "ASDRVIST", "ASRCHKUP", "HPVADSHT",
    "ADPLEASR", "ADDOWN", "ADSLEEP", "ADENERGY", "ADEAT1", "ADFAIL",
    "ADTHINK", "ADMOVE",
)

# Calculated variables where 9 means "don't know / missing".
_CALC_CODES_9 = (
    "_RFHLTH", "_HCVU651", "_RFHYPE5", "_CHOLCHK", "_RFCHOL", "_LTASTH1",
    "_CASTHM1", "_ASTHMS1", "_RACE", "_SMOKER3", "_RFBING5", "_RFDRHV5",
    "_TOTINDA", "PAMISS1_", "_PACAT1", "_PAINDX1", "_PA150R2", "_PA300R2",
    "_PA30021", "_PASTRNG", "_PAREC1", "_PASTAE1", "_LMTACT1", "_LMTWRK1",
    "_LMTSCL1", "_RFSEAT2", "_RFSEAT3", "_FLSHOT6", "_PNEUMO2", "_AIDTST3",
)

MISSING_CODES = {
    **{name: (7, 9) for name in _CODES_7_9},
    **{name: (9,) for name in _CODES_9},
    **{name: (77, 98, 99) for name in _CODES_77_98_99},
    **{name: (9,) for name in _CALC_CODES_9},
    "SCNTWRK1": (97, 98, 99),
    "SCNTLWK1": (97, 98, 99),
    "ASACTLIM": (777, 999),
    "BLDSUGAR": (777, 999),
    "FEETCHK2": (777, 999),
    # Calculated numeric variables with large "missing" sentinels. In the
    # course CSV the implied decimals are already applied, so e.g. the CDC's
    # 99900 for MAXVO2_ appears as 999 and 99000 for PAFREQ1_ appears as 99
    # (checked on x_train.csv). Both versions are listed to be safe.
    "DROCDY3_": (900,),
    "_DRNKWEK": (99900,),
    "MAXVO2_": (999, 99900),
    "FC60_": (999, 99900),
    "PAFREQ1_": (99, 99000),
    "PAFREQ2_": (99, 99000),
    "STRFREQ_": (99, 99000),
    "WTKG3": (99999,),
}

# Codes meaning "none" -> replaced by 0 so the variable becomes a real count.
ZERO_CODES = {
    **{
        name: (88,)
        for name in (
            "PHYSHLTH", "MENTHLTH", "POORHLTH", "CHILDREN", "AVEDRNK2",
            "DRNK3GE5", "DOCTDIAB", "CHKHEMO3", "FEETCHK", "PAINACT2",
            "QLMENTL2", "QLSTRES2", "QLHLTH2", "ASERVIST", "ASDRVIST",
            "ASRCHKUP", "ADPLEASR", "ADDOWN", "ADSLEEP", "ADENERGY", "ADEAT1",
            "ADFAIL", "ADTHINK", "ADMOVE",
        )
    },
    "ASACTLIM": (888,),
}

# 3-digit frequency codes: 1xx = per day, 2xx = per week, 3xx = per month,
# 4xx = per year, 888 = never. Converted to "times per month".
FREQUENCY_COLUMNS = ("BLDSUGAR", "FEETCHK2")

# Remapping of individual codes so that the variable becomes ordinal
# (a larger value = more of the thing). Applied after missing codes.
VALUE_REMAPS = {
    # 1 = yes, 2 = yes but only during pregnancy, 3 = no, 4 = borderline.
    "BPHIGH4": {1: 2, 4: 1, 2: 0, 3: 0},
    # 1 = yes, 2 = only during pregnancy, 3 = no, 4 = pre-diabetes.
    "DIABETE3": {1: 2, 4: 1, 2: 0, 3: 0},
    # 1..4 = within 1/2/5/5+ years, 8 = never -> put "never" after 5+ years.
    "CHECKUP1": {8: 5},
    "EYEEXAM": {8: 5},
    # 8 = never drives or rides in a car -> not meaningful for seatbelt use.
    "SEATBELT": {8: float("nan")},
}

# Nominal categorical variables (numbers are labels, not amounts): one-hot.
NOMINAL_COLUMNS = ("MARITAL", "EMPLOY1", "RENTHOM1", "_RACE")
