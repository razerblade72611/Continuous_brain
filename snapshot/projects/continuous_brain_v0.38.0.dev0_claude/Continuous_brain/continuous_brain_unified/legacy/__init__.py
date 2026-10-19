"""Hash-pinned access to the frozen legacy `continuous_brain` line (C1, DAV-74).

  manifest   VERIFIED entries, root resolution, re-hash verification
  loader     import from the verified root without shadowing
  adapters   step-contract registry and binding (none registered in v0.38.0.dev0)
  inventory  standard-library, read-only inventory that produces UNVERIFIED candidates
"""

from .adapters import (BindingOutcome, StepContract, build_legacy_components, register_step_contract,
                       registered_contracts)
from .loader import LegacyImportConflict, import_verified_module
from .manifest import (ENV_ROOT, MANIFEST_FORMAT, LegacyFile, LegacyHashMismatch, LegacyManifest, ManifestEntry,
                       ManifestError, VerificationResult, parse_entry, resolve_legacy_root, verify_entry)

__all__ = ["BindingOutcome", "ENV_ROOT", "LegacyFile", "LegacyHashMismatch", "LegacyImportConflict", "LegacyManifest",
           "MANIFEST_FORMAT", "ManifestEntry", "ManifestError", "StepContract", "VerificationResult",
           "build_legacy_components", "import_verified_module", "parse_entry", "register_step_contract",
           "registered_contracts", "resolve_legacy_root", "verify_entry"]
