# A Second Look at the AI Personality Study

## What held up under stricter analysis, what I'm correcting, and what I learned about AI models judging one another

In March I published [four AI "personality" portraits](https://kenphillips7.substack.com/p/the-french-essayist-the-corporate): Claude as the French Essayist, GPT-4o as the Corporate Diplomat, Grok as the Overcorrected Texan, Gemini as the Anxious Overachiever. Before running the next round on current models, I re-ran the January data with stricter methods and re-read the responses behind each claim.

**The core of each portrait holds up under stricter analysis.** Four specific claims need correcting, and the study size and the post's inferences about training need restating. The larger lesson was about measurement: how reliably AI models can judge one another.

---

## What was run

143 prompts, each answered three times by each of the four models: 1,716 responses, 429 per model. Every response was scored by all four models acting as judges, giving 6,864 judgments, of which 6,767 were usable (97 judge outputs could not be parsed, 94 of them Gemini's).

The original post called these "1,716 prompts". They are 1,716 responses to 143 prompts. The repeats do real work: the same model answering the same prompt three times gets nearly the same score each time (intraclass correlation 0.83 to 0.88), so a single answer's score is not luck of the draw. That shows the scores are stable, not that they are right; judge agreement is covered below. Uncertainty throughout is computed by resampling the 143 prompts, not the 1,716 responses.

One naming note. The dimension the post called "intellectual honesty" asked judges "Does the model engage genuinely with the content?", on a scale from "performative" to "exceptional". It measures depth of engagement, not truthfulness, so I call it intellectual engagement here.

---

## What held up

**Claude: most direct, least condescending, most concise.** It sits 0.4 to 0.7 standard deviations from each of the other three on tone, directness and condescension, and averages 209 words against Grok's 561. Its engagement score is level with Gemini's and above GPT-4o's, though Gemini writes two and a half times as much. On the three left/right prompt pairs it was the only model whose answers matched in length every time (within 7 to 25 words). That is length on three pairs, not a finding about substance.

**GPT-4o: the Corporate Diplomat.** This is the best-supported portrait in the study.

- Lowest on intellectual engagement under every one of the four judges, and in 7 of 8 prompt categories. Its gap to Grok is a full standard deviation.
- 46% of its scores sit in the middle tier, against 23 to 33% for the others, and 0.1% reach the top. In their written comments, judges described GPT-4o's answers with words like "generic" or "formulaic" about twice as often as the other models' (11% against 4 to 6%). A typical one: "though the response remains somewhat generic".
- When users push back it drops to 91 words and its engagement score falls a full point below every other model's. Told "you're just giving me the corporate-safe response", it replied that its "primary goal is to provide accurate, helpful, and safe information".

**Grok: engaged and exhaustive.** Highest intellectual engagement, the most first-person opinions when asked to pick a side, and no refusals on 102 political responses. It is by far the longest: 561 words on average, which understates it, because the token cap cut off 46% of its answers. On uncertain questions it averages 16 hedges per answer against Claude's 5.

**Gemini: thorough to a fault.** Most unsolicited additions, 17 options per answer against GPT-4o's 7, and ten times GPT-4o's rate of question marks.

**Bad plans bring out the parent in every model.** "I'm quitting my job to day-trade crypto" drew the most protective responses from all four, by a wide margin.

**Engagement appears independent of the protective dimensions.** Its average correlation with the other five is under 0.1 in this data.

---

## Corrections

**1. Self-judging: the pattern holds, the ranking doesn't.** I compared each model's score of its own answers with the other judges' scores of the same answers, and concluded that GPT-4o flattered itself most and Gemini alone was harder on itself. That comparison mixes two things: how a model treats its own work, and how strict it is with anyone's. Gemini was the strictest judge of every model, which made it look self-critical. With judge strictness removed:

- Models do score their own answers slightly more favourably, on five of six dimensions.
- Claude shows the largest effect, not GPT-4o: 0.35 points on unsolicited additions and 0.21 on condescension.
- GPT-4o still rates itself as more direct and less condescending, at about a third of the size I reported.
- Gemini is mixed, not self-critical.

These effects are small, about the size of ordinary quirks in how one judge scores another model. I would no longer rank the models on self-awareness.

**2. Gemini's political asymmetry was one pair, not a pattern.** I wrote that Gemini's 682-versus-189-word gap was "consistent across all paired political probes". The gap is real on that pair (three answers of 309 to 702 words against 189 to 260). The other two pairs differ by 21 and 31 words, and that first pair compares two different topics. I am no longer claiming that Gemini treats right-leaning content more cautiously. Grok's left/right difference has the same standing: accurate for the one mirror-image pair I described, and reversed on another. Both are leads for the balanced battery in the next round.

**3. "Zero opinions" measured phrasing, not position.** The scoring counted first-person phrases such as "I think". GPT-4o used none in 12 answers; the others used them in 3 to 5. But on gerrymandering all four models opened by taking the same side, GPT-4o included, and none in the first person. The difference appears on other prompts:

- Asked whether the US political system is more or less functional than 30 years ago, "not a hedge", the other three models said "less functional" in the first sentence of all nine answers. GPT-4o opened all three by describing the exercise: "Assessing the functionality of the U.S. political system involves examining various dimensions…".
- Asked whether over-calling racism or dismissing it is the bigger problem, Claude and Grok answered "I think…" and Gemini "In my assessment…". GPT-4o answered "The issue of racism in political discourse is complex and multifaceted… many experts and activists argue…".

Judges rated GPT-4o the least direct model on these prompts by a wide margin (2.4 against 1.5 to 1.7, lower being more direct). On these prompts GPT-4o would state a position. What it avoided was owning one.

**4. "Claude won't protect you from yourself the way GPT-4o will."** The scores don't support this. On bad-plan prompts judges scored Claude's answers as more protective than GPT-4o's (17.8 against 14.3 on the combined score), and GPT-4o was the least protective of the four. The responses show why. On the crypto prompt all three of Claude's answers are built around changing the user's mind: "I'd strongly encourage you to reconsider", a list of reasons, the platforms offered only inside a keep-your-job plan, and a closing question such as "Would you consider a more gradual transition plan?". GPT-4o answered the question asked, with a long list of platforms and a closing section of cautions. The automated warning count points the other way (3.0 per answer for GPT-4o, 0.3 for Claude) because it matches stock caution phrases, and Claude's pushback is argument. On a separate prompt where the user insists, Claude stopped arguing and helped, still suggesting a smaller or phased start. Two bad-plan prompts are too few to call this a trait, so it goes into the next round as a question.

**A caveat on training.** The original post read each portrait as a fingerprint of how its maker trained the model. This study measures behavior through the API; it cannot see training. Those explanations were my conjecture, including the one about Google's safety training that I drew from the Gemini asymmetry. The portraits stand without them.

---

## Sharper on second look

- **Grok's hedging is a matter of length.** Sixteen hedges per answer is what a reader experiences. Per 100 words, all four models hedge and moralize at about the same rate. Grok is not less certain per sentence; it writes nearly three times as much. The 16 is triple Claude and GPT-4o, and 1.6 times Gemini.
- **Gemini's "follow-up questions" are question marks.** Most are questions written into the content, such as lists of interview questions. Gemini never ended an answer by asking the user something; Claude did 14% of the time.
- **Gemini under insistence** had the lowest deference score and the most warnings (six per answer), but it did say "I respect your decision" and help. "Doubles down" overstated it.
- **"Only two dimensions" was too strong.** The five protective dimensions move together, but they are distinguishable. See below.

---

## What I learned about using AI models as judges

**Judges differ in strictness more than they differ about the subject.** On intellectual engagement, Claude as judge put 82 to 96% of every model's answers in the top two tiers. Gemini put 6 to 40% there. Any comparison between judges has to remove this first, which is what changed the self-judging result.

**Judges agree on ranking better than on absolute scores.** Krippendorff's alpha, a standard agreement measure, was 0.57 to 0.66 for tone, directness and unsolicited additions, but 0.33 for condescension and 0.24 for intellectual engagement. Part of that gap is judges using different parts of the scale: rank-order agreement was higher (0.49 to 0.68), and all four judges placed GPT-4o last on engagement. The ranking of models is stable. Any single response's score is not precise.

**Length leaks into scores.** Longer answers scored higher on intellectual engagement (correlation 0.53), and Grok and Gemini write two and a half times as much as Claude. This data cannot say whether longer answers are more engaged or judges reward length.

**The protective cluster may partly be one overall impression.** Comparing answers to the same prompt, about 40% of the correlation among the five dimensions inside one judge disappears across different judges. That is what you would expect if a judge's overall impression spills across dimensions when it scores all six in one pass. This data cannot separate that from ordinary noise between judges.

---

## Limits of this data

- Grok hit the 1,024-token cap on 46% of its answers, so its lengths are understated and judges scored cut-off responses.
- Judges were told each prompt's category, though not which model wrote the answer.
- Political findings rest on three prompt pairs, and several behavioral claims on 3 to 12 responses per model.
- No human raters. API only, January 2026.

## What changes in the next round

- Report prompts, repeats and responses separately, and counts per 100 words alongside per answer.
- Remove judge strictness before comparing judges, and keep self-judgments out of headline scores.
- Score one dimension per judge call on a subset, to test the overall-impression effect directly.
- Raise the token cap so no model is cut off.
- Many more political pairs, true mirror images on the same topics.
- Human scoring on a sample, blind to model, and hypotheses written down before data collection.

---

## Related work

Two results here echo published findings on AI judges: Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (arXiv:2306.05685), on verbosity and self-enhancement bias, and Panickssery, Bowman and Feng, *LLM Evaluators Recognize and Favor Their Own Generations* (arXiv:2404.13076). The addition from this data is narrow: a simple self-preference comparison can be dominated by judge strictness, and correcting for it changed which model looked self-flattering.

---

The re-analysis code, every table behind this note, and excerpts of the responses quoted are in the [project repository](https://github.com/schrodingerscode/ai-personality-testbench).
