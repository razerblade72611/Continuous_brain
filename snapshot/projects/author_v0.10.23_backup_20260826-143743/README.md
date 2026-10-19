# AUTHOR v0.10.23 — Carried Destination Probe

AUTHOR v0.10.23 preserves the semantic-planner/private-motor separation and the stable Level-4 placement contract while fixing the final two v0.10.22 place-route failures.

The trainer now physically preflights the held-object route from the far-side door landing to the exact pinned placement pose using Unity collision authority, and the runtime can reuse a trainer-preflighted pinned terminal when a later reachable/interactable refresh omits that exact grid cell after the target component is already visible.

See `AUTHOR_v0.10.23_OPERATOR_README.md` for upgrade, repair, and qualification commands.
