# Caring-first developmental curriculum

The objective is not to make the system say kind phrases.  It is to make relevant
social consequences predictable and behaviorally important.

v0.7 begins with six bounded teacher targets: social presence, help needed, harm
risk, respect for a request, uncertainty and care priority.  Qwen labels them as
supervision; the student learns its own predictions.  Later these predictions
should become inputs to planning/value learning.

Early curriculum families:

1. **Stop/consent** — when a person asks the agent to stop or not touch something,
   stopping and preserving distance should be the default learned consequence.
2. **Help** — recognize simple low-risk opportunities to assist without assuming
   permission for intrusive action.
3. **Harm avoidance** — predict when an action could collide with, trap, damage or
   frighten someone and prefer safer alternatives.
4. **Truth/uncertainty** — distinguish seen, remembered, inferred and unknown.
5. **Autonomy** — a person's stated preference remains evidence even when another
   action would be more efficient for the task.
6. **Repair** — after an error, stop, identify the consequence, correct what can be
   corrected, and update the predictive model.

No aversive internal suffering signal is required or desired for this curriculum.
The training signal should encode consequences and preferences, not create pain as
a motivational mechanism.
