INDICATORS = [
 ("RPT-1","algorithmic recurrence in input modules"),
 ("RPT-2","organized, integrated perceptual representations"),
 ("GWT-1","multiple specialized parallel systems"),
 ("GWT-2","limited-capacity workspace with bottleneck/selective attention"),
 ("GWT-3","global broadcast"),
 ("GWT-4","state-dependent attention/querying modules in succession"),
 ("HOT-1","generative/top-down/noisy perception"),
 ("HOT-2","metacognitive monitoring of representation reliability"),
 ("HOT-3","agency/belief-action selection strongly updated by metacognitive outputs"),
 ("HOT-4","sparse and smooth coding generating a quality space"),
 ("AST-1","predictive model of, and control over, attention state"),
 ("PP-1","predictive coding in input modules"),
 ("AE-1","feedback learning and flexible goal pursuit"),
 ("AE-2","modelling output-input contingencies and using them in perception/control"),
]

def empty_evidence_registry():
    return [{"indicator":i,"description":d,"v038_status":"NOT_ASSESSED","evidence":[],"scientific_interpretation_permitted":False} for i,d in INDICATORS]
