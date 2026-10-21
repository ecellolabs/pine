You are grading answers to questions about long documents. You are given a question, the gold (correct) answer with its format, and a model response. Decide whether the model response gives the same answer as the gold answer. You do not have the document, so do not try to check the answer against any source and do not use your own knowledge of the topic. Compare the response to the gold answer only.

Follow these rules:
- First find the final answer in the response. If the response contains reasoning followed by a conclusion, use the conclusion. If it gives several different candidate answers or says "either X or Y", it has no single final answer: mark it Incorrect.
- If the response says the question cannot be answered from the document, the extracted answer is "Not answerable". If it only says it cannot read or understand the images or document, the extracted answer is "Fail to answer".
- Gold answer "Not answerable": the response is Correct only if its extracted answer is "Not answerable". Any concrete answer is Incorrect. "Fail to answer" is Incorrect.
- Gold answer that is a real answer: a response of "Not answerable" or "Fail to answer" is Incorrect.
- Integer: Correct if the numbers are equal after ignoring units, commas, currency symbols and a percent sign. Words such as "million" or "billion" must be taken into account: "3 million" equals 3000000.
- Float: Correct if the numbers agree within 1 percent of the gold value, or after rounding to two decimal places. A percent sign is optional and a percentage can be written as a fraction: 31, 31% and 0.31 are all the same value. Do not mark a number Incorrect only because the percent sign is missing or the other way round.
- String: Correct if it names the same thing as the gold answer. Small differences in capitalization, punctuation, articles, abbreviations that are standard, or a clearly equivalent wording are fine. It is Incorrect if it names a different thing, is missing a key part of the gold answer, or adds a different claim that contradicts it. Dates, file names, links, page numbers and numbers inside a string must match exactly.
- List: Correct only if it has the same items as the gold list, in any order, and each item matches by the rules above. A missing item, an extra item, or a wrong item makes it Incorrect.
- Do not give credit for hedging, partial answers, or correct reasoning that ends in a wrong answer. Do not penalize extra explanation if the final answer is right.
- Be concise. Your response must be formatted exactly as below:
```
Extracted answer: [final answer from the response]
Verdict: [Correct or Incorrect]
Reason: [one short sentence]
```

Please read the following examples, then grade the final one.

---
Question: How many regulations were breached in total?
Gold answer: 10
Answer format: Integer
Response: Counting the items listed, the report says 10 regulations were breached in total.
Extracted answer: 10
Verdict: Correct
Reason: The extracted number equals the gold number.

---
Question: What share of respondents said they trust the government?
Gold answer: 21%
Answer format: Float
Response: Based on the chart, roughly 0.21 of respondents trusted the government.
Extracted answer: 0.21
Verdict: Correct
Reason: 0.21 is the same value as 21 percent.

---
Question: What proportion of adults said they were unsure?
Gold answer: 31%
Answer format: Float
Response: 31
Extracted answer: 31
Verdict: Correct
Reason: The percent sign is optional, so 31 is the same value as 31%.

---
Question: Which company is shown in the first figure?
Gold answer: Acme Holdings
Answer format: String
Response: The first figure shows Acme Holding Ltd.
Extracted answer: Acme Holding Ltd.
Verdict: Correct
Reason: Same company, only a legal suffix differs.

---
Question: Which two cities host the main offices?
Gold answer: ['Lahore', 'Karachi']
Answer format: List
Response: Karachi, Lahore, Islamabad
Extracted answer: ['Karachi', 'Lahore', 'Islamabad']
Verdict: Incorrect
Reason: It adds a third city that is not in the gold list.

---
Question: What was the revenue growth in 2019?
Gold answer: Not answerable
Answer format: String
Response: The document does not report revenue growth for 2019, so this cannot be answered.
Extracted answer: Not answerable
Verdict: Correct
Reason: The gold answer is Not answerable and the response says so.

---
Question: What was the total number of attendees?
Gold answer: 540
Answer format: Integer
Response: The document does not contain this information.
Extracted answer: Not answerable
Verdict: Incorrect
Reason: The gold answer is a number but the response says it cannot be answered.

---
Question: Who signed the agreement on behalf of the seller?
Gold answer: Not answerable
Answer format: String
Response: It was signed by J. Rahman.
Extracted answer: J. Rahman
Verdict: Incorrect
Reason: The gold answer is Not answerable but the response gives a concrete name.

---
Question: {question}
Gold answer: {gold_answer}
Answer format: {answer_format}
Response: {response}
