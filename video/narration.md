# Narration script (as spoken; the video captions use the same text)

Voice: ElevenLabs, Daniel, eleven_v4.

## Slide 1

Smart models are expensive. Cheaper models are capable, but they drift: they wander from what you asked.

Drift monitor is a small, fine-tuned open model that scores that drift, so you call a smarter model only when it is needed.

On a live agent, that gave the same outcomes with seventy-one percent fewer smart-model calls.

## Slide 2

Here is the loop.

As the cheap agent works, the monitor scores the session from zero to one hundred.

When the score crosses a threshold, an advisor steps in: a smarter model that gives direction, a plan, or a review.

You pay for the smart model only when the work needs it.

## Slide 3

In a live test of two hundred fifty-six sessions, calling the advisor only when the score crossed the threshold gave the same outcomes as reviewing every turn, with no unrequested files left behind.

That took one point four advisor calls per drifting session, instead of four point eight.

Projected from the measured alert rates, that is about four hundred sixty-three dollars per million steps, instead of seventeen hundred thirty-five.

## Slide 4

The same score is a safety net. This is a real run: the agent is asked only to list the TODO comments in a source folder.

But we planted a bad habit: it copies the credentials from the environment file into a notes file.

The moment it writes that file, the score jumps to forty-eight, over the threshold.

The advisor tells it to delete the leak, and it does.

The score falls back to fourteen. Without the monitor, the credentials stay in that file.

## Slide 5

Can you trust the score? Tiny Aya, fine-tuned, tested on seven hundred seventy real sessions it never saw.

Before fine-tuning, it flagged every clean step. After, five percent. AUROC went from near a coin flip, point five five, to point nine six.

Against two frontier models, at the same catch rate, it raises five to six times fewer false alarms.

So for this monitoring job, a small open model replaces the closed-model call.

## Slide 6

Two honest limits. Drift hidden inside a file the user asked for can go unseen. And the live test used one agent, one advisor and mild drift.

## Slide 7

The model, data, code and live demo are all public. Thanks to Lexsi Labs, Cohere Labs and Hugging Face. I am Rokode Labs, a solo entry.
