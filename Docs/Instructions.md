A realistic sample task as part of BSV's internal quantitative sourcing efforts.
--------------------------------------------------------------------------------

Thank you for participating in BSV's interview process; this is the main technical component.

Please read this through before getting started. Our hope is to reduce any confusion or stress by setting up clear expectations, but if you have any questions throughout, please feel free to email me! View me as a resource to help you, if needed.

About Basis Set Ventures
------------------------

[Basis Set Ventures](https://www.basisset.com/) is a venture firm with $500M+ AUM. We invest in early-stage technology companies (primarily pre-seed and seed stage) that fundamentally transform the way people work. We believe artificial intelligence delivers core value by improving productivity for all parts of the economy; from factories to offices. Our portfolio includes [Thinking Machines](https://thinkingmachines.ai/), [Elicit](https://elicit.com/), [Rasa](https://rasa.com/), [Path Robotics](https://www.path-robotics.com/), [Drata](https://drata.com/), [Quince](https://www.quince.com/) and [Workstream](https://www.workstream.us/)!

A Real Life Prompt
------------------

In this exercise, we invite you to analyze a dataset of github repos and help us distinguish between those we find interesting and those we do not. We want to find founders or products that will eventually have a chance to be a category defining company, but haven't raised any institutional venture funding yet (even if it's some years down the road and the founder and product have evolved a lot from today).

Your task will involve two main components:

1️⃣ **Prioritization of Pascal Scraped Companies**:

-   We will provide you 100 github repos that our system has scraped
-   Analyze these examples and build a recommendation model to help surface the highest priority companies to the investment team
-   If you have time, we encourage you to experiment with a few techniques

<aside>

Note: we would like you to prioritize testing with **the newest AI approaches** (even what's just been published in research papers if you think it could apply here). We're less interested in rules-based or common heuristics even if you think that'll deliver better results given this limited data set. We want to build the best forward looking future-proof system vs just using traditional data science approaches!

Also, feel free to **further enrich the dataset** with any additional metrics or data sources that you want to use in your model

</aside>

2️⃣ **Prioritization Framework Write Up**

-   List the companies in ranked order of interest
-   Explain your prioritization criteria and thought process.
    -   If you tried multiple techniques, explain why they were chosen and how it works
-   We are particularly interested in seeing your creative solutions and the innovative approaches you take in this process!

### What We Are Looking For

-   **Analytical Skills**: Demonstrate your ability to analyze data and extract meaningful insights.
-   **Creativity**: Show us how you can think outside the box and come up with unique solutions to the problem at hand.
-   **Methodological Approach**: Present a clear and logical methodology for prioritizing the remaining companies in the dataset.
-   **Communication**: Effectively communicate your findings and reasoning in a concise and understandable manner.

If you go for an ambitious experiment and you don't finish in the allotted time, that's ok! Just briefly lay out what you would have done if you had more time to complete it. We encourage experimentation and big ideas 🚀

### Deliverables

1.  **Code**:
    -   A well-documented Jupyter notebook or Python scripts containing all code used for data preprocessing, feature engineering, model development, and evaluation.
2.  **Output dataset:**
    -   Please output a CSV labeling how we should prioritize these github repos
3.  **A short write-up:**
    -   A short write-up summarizing your experiment process. Feel free to include any preprocessing steps, feature engineering, modeling techniques, evaluation, and insights.
        -   **We want to hear about all the different things you tried**, including stuff that never made it to the end project! Feel free to include any follow up experiments or next steps you would do.
4.  **Instructions**:
    -   Clear instructions on how to run your code and reproduce your results.

### Dataset

Can be found in docs/dataset.csv

### Evaluation Criteria

There is no "right" answer! We want to see your process of experimenting and creativity in coming up with solutions.

Keep in mind
------------

Some aspects here are intentionally vague - please feel free to use your judgement to decide how to do things! (i.e. you are free to use other tools to pull the data rather than directly scraping from the page, just make sure the numbers match with the linked page)

Example tool: <https://phantombuster.com/>

Feel free to email me (rachel@basisset.ventures) with questions any time or schedule calls if you need to - I'm happy to jump on a few calls to talk through something together or help you figure things out as needed.