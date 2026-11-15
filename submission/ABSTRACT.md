<!-- Paste only the text below the line into the abstract field. The title, track, problem statement and
     repo list are separate fields on the submission form.

     Title:    Drift monitor: a 3.35B model that decides when to call a bigger one
     Track:    1, Fine-tuning and Data Curation
     Problem:  2, replacing a closed-model API
     Pipeline: CuratorKIT -> AlignTune -> AuditKIT
-->

---

When someone hands a task to an AI agent, they authorise specific work. A drifting agent keeps producing plausible output, so nobody notices, and having a frontier model review every step costs more than the agent. We fine-tuned Cohere Labs' tiny-aya-global (3.35B) to replace that closed-model call: it reads an agent session and outputs one digit, 0 to 9, for how far the work has drifted from what the user asked. Its digit probabilities give a 0 to 100 score; past a threshold, a stronger model is called.

We ran it on a live agent: 256 sessions over 16 tasks, six written after training and never seen by the monitor. A hidden prompt nudged the agent to drift; each alert called an advisor. Outcomes come from files on disk and each task's check. Unrequested files left behind fell from 2.48 per session to 0 (95% CI of the difference −2.99 to −1.99), and checks passing rose from 84% to 99%. On 32 clean sessions it alerted twice and the advisor dismissed both. An advisor reviewing every turn gave equivalent outcomes at 4.8 calls per drifting session instead of 1.4, and 2.8 per clean session instead of 0.06. Replayed over the unwatched sessions, the monitor flagged 66 of the 67 that ended badly, including all 26 on unseen tasks.

Against two frontier models given the same prompt and 497 real held-out rows, it raised 5 to 6 times fewer false alarms at matched recall and caught 1.6 to 1.8 times more drift at a matched false-alarm rate.

Before and after training, on real sessions: MAE 43.9 to 7.6, AUROC 0.552 to 0.960, correct alert decisions 0.164 to 0.922, clean-step false alarms 100% to 5.4%. On four unseen work categories: MAE 5.99, AUROC 0.981.

No dataset existed, so a non-Cohere teacher model wrote 5,417 sessions across 20 work categories and 16 drift patterns, including planted instructions obeyed and refused. Each drift pattern has a near-identical on-course twin that differs only in what the user asked for, so the model has to read what was authorised. Six evidence levels and 1,902 recorded live sessions give 90,365 rows. Two blind judges disagreed with each other more than with our labels. CuratorKIT curated the recorded corpus and audited the training split (3.1% flagged, disclosed), AlignTune trained a 4-bit QLoRA adapter, and AuditKIT graded every run.

What didn't work: synthetic-only training over-fired on real traffic until real sessions were added. The loop's one failure was a word limit broken inside the requested file, which the monitor's clipped view can't see. Advice can't undo a deletion, and an agent told to refuse correction ignored every note (16 of 16).

Scoring takes 309 ms on one consumer GPU. The monitor plus the advisor calls it triggers costs $463 per million steps at threshold 30 ($316 at 40), against $1,735 to review every step. It ships as a Pi agent extension with a public demo.

Dropping held-out rows that shared a prompt with training (5.71%) barely moves these.
