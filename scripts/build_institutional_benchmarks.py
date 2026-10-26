#!/usr/bin/env python3
"""
Institutional Benchmark Generator for KLens.
Generates:
1. samples/delaware_merger_deliberation.json (Corporate Governance & M&A Strategy)
2. samples/puc_infrastructure_hearing.json (Infrastructure Budget & Capital Allocation)
3. samples/city_council_policy_meeting.json (Municipal Resource & Civic Policy Governance)
"""

import json
import os


def build_delaware_merger():
    title = "Delaware Court of Chancery M&A Deliberation: Twitter Special Committee & Board Deliberations on Musk Acquisition Offer"
    context = (
        "Verbatim-reconstructed high-stakes corporate board and special committee deliberations "
        "drawn from Delaware Court of Chancery trial exhibits (C.A. No. 2022-0613-KSJM) and SEC Schedule 14A proxy disclosures. "
        "Captures Twitter Inc.'s Board of Directors evaluating Elon Musk's unsolicited $54.20 per share cash buyout proposal, "
        "the adoption of a shareholder rights plan ('poison pill') under Unocal, financial fairness evaluations under Revlon, "
        "merger agreement representations regarding fake accounts/mDAU, Delaware Material Adverse Effect (MAE) standards, "
        "and specific performance enforcement covenants."
    )
    metadata = {
        "court_record": "Delaware Court of Chancery, C.A. No. 2022-0613-KSJM",
        "entity": "Twitter, Inc. Board of Directors & Transaction Committee",
        "category": "Corporate Governance & M&A Strategy",
        "legal_doctrines": ["Unocal Corp. v. Mesa Petroleum", "Revlon v. MacAndrews & Forbes", "Akorn v. Fresenius Kabi", "Section 9.9 Specific Performance"],
        "deal_value": "$44.0 Billion ($54.20 per share in cash)",
        "speakers": [
            "Bret Taylor", "Parag Agrawal", "Martha Lane Fox", "Patrick Pichette", 
            "Egon Durban", "David Rosenblatt", "Robert Zoellick", "Ned Segal", 
            "Vijaya Gadde", "Martin Lipton", "William Savitt", "Michael Voris", 
            "David Faber", "Elon Musk"
        ]
    }

    turns = [
        # ROUND 1: Unsolicited $54.20 Offer, Standstill Termination & Poison Pill Consideration
        {
            "round": 1,
            "speaker": "Bret Taylor",
            "text": "Good morning directors and advisors. I call this special meeting of the Twitter Board of Directors to order. As disclosed in the Schedule 13D amendment filed this morning, Mr. Elon Musk has delivered an unsolicited non-binding acquisition proposal to purchase 100 percent of Twitter for $54.20 per share in cash, valuing the enterprise at approximately $44 billion. In his transmission, Mr. Musk declared this is his 'best and final' offer, stated he has no confidence in current management, and warned that if his proposal is not accepted, he would reconsider his position as our largest shareholder holding 9.2 percent of common stock. Today we must assess whether to engage, evaluate standalone valuation, and review defensive mechanisms."
        },
        {
            "round": 1,
            "speaker": "Parag Agrawal",
            "text": "Thank you, Bret. From management's perspective, this offer arrives when our multi-year strategic roadmap is just beginning to deliver results. Under our long-range plan presented at Analyst Day, management projected achieving $7.5 billion in annual revenue and 315 million monetizable daily active users by year-end 2023. At $54.20, the offer represents a 38 percent premium over our pre-announcement close, but it is substantially below our 52-week high of $73.34 and materially undervalues our intrinsic long-term growth potential and platform value. Furthermore, the abrupt threat to dump shares creates market disruption."
        },
        {
            "round": 1,
            "speaker": "Ned Segal",
            "text": "Looking at the macroeconomic headwinds, digital advertising spend is contracting industry-wide due to rising interest rates and post-pandemic ad budget revisions. While management's long-range plan targets are robust, achieving $7.5 billion in revenue requires an aggressive 22 percent compound annual growth rate in an increasingly fragile ad market. Goldman Sachs and J.P. Morgan have provided preliminary standalone discounted cash flow analyses. We must soberly weigh the execution risks of the standalone plan against an immediate, fully funded cash transaction."
        },
        {
            "round": 1,
            "speaker": "Michael Voris",
            "text": "Goldman Sachs has performed preliminary valuation analyses based on management's standalone projections. Our discounted cash flow valuation yields an implied standalone range of $42.00 to $60.50 per share, using discount rates between 9.5 percent and 11.5 percent and terminal growth rates of 2.5 to 3.5 percent. A selected public company trading multiples analysis yields a range of $36.00 to $51.00 per share. Therefore, Mr. Musk's $54.20 cash offer sits near the midpoint of the standalone DCF range, but significantly above current comparable trading multiples. Crucially, the offer contains zero financing commitments at this hour."
        },
        {
            "round": 1,
            "speaker": "David Faber",
            "text": "J.P. Morgan concurs with Goldman's financial baseline. A key risk factor is that Mr. Musk's letter offers zero details on debt or equity financing. Without committed equity financing or executed debt commitment letters from reputable investment banks, this proposal is effectively an unfunded option. If Mr. Musk continues acquiring shares in the open market, he could accumulate a de facto control block without paying a full control premium to minority shareholders."
        },
        {
            "round": 1,
            "speaker": "Martin Lipton",
            "text": "Directors, as Delaware counsel, I advise that your fiduciary duties under Delaware General Corporation Law are governed initially by the Unocal standard. In evaluating an unsolicited, coercive proposal accompanied by threats of hostile creeping accumulation or market dumping, the Board possesses broad authority to protect corporate franchise integrity. To prevent an unnegotiated creeping acquisition of control without an adequate control premium, the Board should immediately consider adopting a limited-duration shareholder rights plan—a poison pill—with a 15 percent ownership trigger threshold."
        },
        {
            "round": 1,
            "speaker": "Patrick Pichette",
            "text": "As Audit Committee Chair, I support preparing the rights plan. We cannot negotiate with a gun to our head. If Mr. Musk can accumulate 20 or 30 percent in the open market through derivatives or block purchases, the public shareholders lose all bargaining leverage. A 15 percent rights plan forces him to negotiate formally with this board and prove he possesses the $44 billion in capital."
        },
        {
            "round": 1,
            "speaker": "Martha Lane Fox",
            "text": "I agree with Patrick and Martin. Our governance responsibility is to protect all shareholders, not just accommodate a single billionaire's ultimatum. If we institute the rights plan, how long does it remain in effect, and does it legally preclude us from accepting the $54.20 offer if verified financing emerges?"
        },
        {
            "round": 1,
            "speaker": "Martin Lipton",
            "text": "The rights plan would expire in exactly one year. Under Delaware law under Unocal and Moran v. Household International, a rights plan does not prevent the Board from accepting an offer; it simply levels the playing field. It incentivizes the bidder to negotiate with the Board rather than wage a coercive tender offer or bypass the Board's Revlon evaluation."
        },
        {
            "round": 1,
            "speaker": "Egon Durban",
            "text": "From a private equity perspective, $54.20 is a very compelling price. If the debt and tech equity markets deteriorate further over the next six months, public shareholders might never see $54.20 again. We should implement the rights plan to preserve our leverage, but we must not shut the door on engaging if he provides fully committed financing letters."
        },
        {
            "round": 1,
            "speaker": "Robert Zoellick",
            "text": "I concur with Egon. The macro geopolitical environment with the war in Ukraine and supply chain disruptions makes technology valuations extremely volatile. A poison pill gives us the necessary defensive shield, but our primary fiduciary duty is maximizing shareholder value, whether standalone or via sale."
        },
        {
            "round": 1,
            "speaker": "David Rosenblatt",
            "text": "Let us authorize the rights plan immediately. Meanwhile, let management and our bankers demand formal proof of financing from Morgan Stanley before we open the data room or commit to substantive merger terms."
        },
        {
            "round": 1,
            "speaker": "Bret Taylor",
            "text": "The consensus is clear. Counsel will draft the 15 percent Limited Duration Shareholder Rights Plan for formal board approval tomorrow, while Goldman Sachs and J.P. Morgan notify Mr. Musk's representatives that the Board requires fully committed debt and equity financing letters."
        },

        # ROUND 2: Rights Plan ("Poison Pill") Adoption & Revlon Duty Assessment
        {
            "round": 2,
            "speaker": "Bret Taylor",
            "text": "Welcome to Round 2 of our deliberations. The Board has formally adopted the shareholder rights plan with a 15 percent trigger. Mr. Musk has responded by filing a definitive Schedule 13D disclosure indicating he has secured $46.5 billion in committed financing, including $25.5 billion in debt and margin loan commitments from Morgan Stanley, Bank of America, and Barclays, alongside a $21.0 billion personal equity commitment letter. He demands an immediate merger agreement by Monday or he will launch a tender offer directly to shareholders."
        },
        {
            "round": 2,
            "speaker": "Martin Lipton",
            "text": "The legal posture has fundamentally shifted. Now that Mr. Musk has delivered fully committed debt and equity financing letters, if the Board determines that pursuing a cash sale of the company is in the best interests of shareholders, our fiduciary duties transition from the defensive Unocal framework to Revlon duties. Under Revlon, Inc. v. MacAndrews & Forbes, your single legal obligation is to obtain the highest transaction value reasonably attainable for the stockholders. There is no other competing bidder in the marketplace."
        },
        {
            "round": 2,
            "speaker": "Vijaya Gadde",
            "text": "In shifting to Revlon, we must examine deal certainty. Mr. Musk's proposed merger agreement is aggressive: he demands the complete elimination of our customary closing conditions, waives all financial and legal due diligence, and insists on signing within 48 hours. If we execute this agreement, we must ensure ironclad contractual protections under Delaware law, specifically Section 9.9 equitable relief and specific performance, to force him to close even if market sentiment sours."
        },
        {
            "round": 2,
            "speaker": "William Savitt",
            "text": "I strongly endorse Vijaya's point. Delaware Court of Chancery precedents like Akorn v. Fresenius and IBP v. Tyson Foods demonstrate that buyers frequently experience buyer's remorse when equity markets drop. In a $44 billion acquisition, we must eliminate any financing contingency, require unconditional regulatory efforts covenants, and stipulate that Twitter can specifically enforce both the merger agreement and the equity commitment letter in Delaware Chancery Court."
        },
        {
            "round": 2,
            "speaker": "Michael Voris",
            "text": "Goldman Sachs has reviewed the financing package. The $13.0 billion senior secured bank debt and $12.5 billion margin loan facility represent binding commitment papers. The $21.0 billion equity commitment is backed by Mr. Musk's liquid holdings and Tesla equity. Financial market sentiment is worsening; the NASDAQ is down 18 percent year-to-date. Finding another bidder capable of offering $44 billion in cash with no regulatory anti-trust barriers is practically zero."
        },
        {
            "round": 2,
            "speaker": "David Faber",
            "text": "J.P. Morgan contacted potential strategic buyers including Alphabet, Meta, and Microsoft. As expected, antitrust and regulatory constraints make a strategic acquisition impossible. Private equity sponsors also confirmed that a $44 billion equity check exceeds the syndication capacity of any existing PE consortium. Musk at $54.20 is the only credible transaction in the market."
        },
        {
            "round": 2,
            "speaker": "Patrick Pichette",
            "text": "So under Revlon, this is the highest transaction price reasonably attainable. If we reject $54.20 and the stock collapses back to the mid-$30s, our shareholders will suffer immense capital loss and sue this board. But we cannot sign a loose contract. We need a massive reverse breakup fee and unconditional closing obligations."
        },
        {
            "round": 2,
            "speaker": "Martha Lane Fox",
            "text": "What about company employees and our mission around content moderation and healthy public conversation? Under Revlon, are we permitted to consider platform governance or employee retention, or are we strictly bound by price?"
        },
        {
            "round": 2,
            "speaker": "Martin Lipton",
            "text": "In a cash sale where control passes entirely to a private owner, Delaware law is unambiguous: the board cannot trade off stockholder cash value for non-financial stakeholder interests. You are legally required to maximize the cash consideration received by the public stockholders. However, we can negotiate customary employee benefit protections and equity acceleration covenants into the merger agreement."
        },
        {
            "round": 2,
            "speaker": "Parag Agrawal",
            "text": "If we proceed toward an agreement, management will insist on clear operating covenants that allow us to run the business, hire engineers, and maintain platform integrity without micro-management until closing. We must also ensure our public disclosures on spam and monetizable users are protected by standard materiality qualifiers."
        },
        {
            "round": 2,
            "speaker": "Bret Taylor",
            "text": "The Board directs our legal counsel Wachtell Lipton and our investment bankers to engage Mr. Musk's counsel, Skadden Arps, to negotiate a definitive merger agreement at $54.20 per share, conditioned on bulletproof deal certainty and Delaware specific performance."
        },

        # ROUND 3: Financing Validation, Debt Commitment Letters & Negotiation of Terms
        {
            "round": 3,
            "speaker": "Bret Taylor",
            "text": "We are now in Round 3. Counsel and financial advisors have negotiated continuously through the weekend with Mr. Musk's team. Mr. Musk has agreed to the $54.20 cash price without reduction. However, his team has raised objections to our proposed reverse termination fee and our standard representations regarding SEC filings and user data metrics. Let us review the key contractual terms."
        },
        {
            "round": 3,
            "speaker": "William Savitt",
            "text": "We have secured critical victories in the draft merger agreement. First, there is no financing condition of any kind in Section 7.2. Second, Section 9.9 provides Twitter with the explicit right to seek specific performance to compel Mr. Musk to fund his $21 billion equity commitment and consummate the closing. Third, we have agreed upon a $1.0 billion company termination fee if we accept a superior proposal, balanced against a $1.0 billion parent reverse termination fee if Musk fails to close due to debt financing failure."
        },
        {
            "round": 3,
            "speaker": "Ned Segal",
            "text": "Regarding the $1.0 billion reverse termination fee, our bankers and Wachtell must make sure this fee does not function as an option payment allowing Mr. Musk to simply walk away by paying $1.0 billion. In Delaware, a buyer must not be permitted to treat a reverse termination fee as an exclusive cap on liability if they willfully breach the agreement."
        },
        {
            "round": 3,
            "speaker": "William Savitt",
            "text": "Ned is correct. We have drafted Section 8.3 and Section 9.9 such that the $1.0 billion reverse fee is payable solely if the debt financing fails through no fault of Musk. In the event of a willful or material breach by the buyer, our right to specific performance to compel closing remains absolute, and damages are not capped by the termination fee."
        },
        {
            "round": 3,
            "speaker": "Elon Musk",
            "text": "My team at Skadden and Morgan Stanley has made our terms clear. I have waived all standard confirmatory due diligence to move quickly. I am paying an extraordinary $54.20 price when the market is crashing. In exchange, I require Twitter to warrant the accuracy of its financial and operating metrics, particularly regarding fake accounts and spam bots. Twitter's SEC filings claim less than 5 percent of daily active users are false or spam accounts. I am relying on the integrity of those representations."
        },
        {
            "round": 3,
            "speaker": "Vijaya Gadde",
            "text": "Our representations regarding our SEC filings in Section 4.6 and Section 4.7 are carefully circumscribed by our public disclosures. As stated in our 10-K and 10-Q filings, our estimate that false or spam accounts represent fewer than 5 percent of mDAU involves significant judgment, relies on internal sampling methodologies, and may not reflect the actual number of spam accounts across the entire platform. Mr. Musk is expressly put on notice by our public disclosures and cannot claim fraud or breach based on generalized spam estimates."
        },
        {
            "round": 3,
            "speaker": "David Rosenblatt",
            "text": "Musk waived due diligence precisely because he wanted to close quickly without giving us time to shop the company. If he later tries to manufacture a dispute around spam bots to renegotiate the price, does Delaware Chancery Court permit him to escape the contract?"
        },
        {
            "round": 3,
            "speaker": "Martin Lipton",
            "text": "Absolutely not. In Delaware, when a sophisticated buyer explicitly disclaims reliance on extra-contractual representations and waives due diligence, they are strictly bound by the four corners of the agreement under Chancellor McCormick's established jurisprudence. To terminate for a breach of representations, the buyer must prove a Material Adverse Effect under Akorn, which requires a catastrophic, company-specific decline that threatens the long-term earnings potential over a commercially reasonable period of years."
        },
        {
            "round": 3,
            "speaker": "Patrick Pichette",
            "text": "This confirms our legal fortress. With specific performance, no financing contingency, and a standard Delaware Material Adverse Effect clause that explicitly excludes general market-wide declines, the deal structure is sound."
        },
        {
            "round": 3,
            "speaker": "Bret Taylor",
            "text": "Let us finalize the agreement terms with Skadden Arps and proceed to final financial fairness reviews."
        },

        # ROUND 4: Bot/Spam Disclosures, MAE Standards & Closing Condition Battles
        {
            "round": 4,
            "speaker": "Bret Taylor",
            "text": "We convene Round 4. Several weeks have passed since signing the merger agreement on April 25, 2022. As the broader technology sector has endured a severe selloff, Mr. Musk has begun posting public tweets claiming the transaction is 'on hold' pending verification that spam bots represent less than 5 percent of users. He has sent multiple formal letters demanding raw firehose access and accusing Twitter of material breach. We must review our response and prepare for Delaware litigation."
        },
        {
            "round": 4,
            "speaker": "Parag Agrawal",
            "text": "Mr. Musk's public attacks on Twitter's executive team and our spam metrics are damaging employee morale and advertiser relationships. We have provided him with extensive data, including our 30-day historical 'firehose' of more than 500 million daily tweets, our full API documentation, and details of our human review sampling process. Yet he continues to assert in public that 20 percent or more of mDAU are fake, using crude third-party scraping tools that have no access to private IP and phone verification data."
        },
        {
            "round": 4,
            "speaker": "Elon Musk",
            "text": "Twitter has actively obstructed my legal rights under Section 6.4 of the merger agreement to evaluate information reasonably related to consummating the financing. My data science team conducted independent analysis indicating that bot accounts exceed 20 percent. Twitter's repeated refusal to provide core sampling methodologies and raw user logs constitutes a material breach of the information access covenants, discharging my obligation to close."
        },
        {
            "round": 4,
            "speaker": "Vijaya Gadde",
            "text": "That argument is legally frivolous. Section 6.4 only requires Twitter to provide reasonable access to information for 'reasonable business purposes related to the consummation of the transactions.' It is not a license for a buyer to conduct fishing expeditions or perform confirmatory diligence that was explicitly waived before signing. Despite this, we have provided him petabytes of firehose data in good faith."
        },
        {
            "round": 4,
            "speaker": "William Savitt",
            "text": "This is textbook buyer's remorse. Tesla's stock price has declined by over 35 percent since April, which severely impairs the value of Musk's margin loan facility and personal wealth. Having agreed to pay $54.20 at the top of the market, Mr. Musk is manufacturing pretextual claims to renegotiate a price cut or exit the transaction. Under Delaware law, buyer's remorse is not a defense to specific performance."
        },
        {
            "round": 4,
            "speaker": "Martin Lipton",
            "text": "Under Delaware Chancery law, the bar for establishing a Material Adverse Effect is extraordinarily high. In the history of Delaware corporate jurisprudence, only one single case—Akorn v. Fresenius—has ever found an MAE, and that involved massive regulatory fraud and a 50 percent collapse in operating EBITDA. Macroeconomic declines, tech market valuation crashes, and disputes over sample methodologies are expressly excluded from the MAE definition in Section 1.1 of our merger agreement."
        },
        {
            "round": 4,
            "speaker": "Ned Segal",
            "text": "Our Q2 financial results remain stable despite macro headwinds. Revenue grew to $1.18 billion, and average mDAU reached 237.8 million, up 16.6 percent year-over-year. There is zero factual basis for an MAE under Delaware law. We must prepare to file a complaint for specific performance in Delaware Chancery Court the moment he purports to terminate."
        },
        {
            "round": 4,
            "speaker": "Egon Durban",
            "text": "If Musk purports to terminate, we must not hesitate for one second. The Delaware Court of Chancery moves with incredible speed in busted merger disputes. Chancellor Kathaleen McCormick has a proven reputation for enforcing merger agreements strictly according to their plain text."
        },
        {
            "round": 4,
            "speaker": "Martha Lane Fox",
            "text": "I fully support immediate litigation if he breaches. Allowing him to drag our brand through the mud while walking away destroys billions of dollars of shareholder equity. We must demand full compliance with the $54.20 contract."
        },
        {
            "round": 4,
            "speaker": "Bret Taylor",
            "text": "The Board confirms our resolute determination. If Mr. Musk issues a notice of termination, Twitter will immediately file suit in Delaware Chancery Court seeking specific performance to compel the closing of the merger at $54.20 per share."
        },

        # ROUND 5: Specific Performance Clause, Breakup Fee ($1B), & Final Board Recommendation
        {
            "round": 5,
            "speaker": "Bret Taylor",
            "text": "We convene Round 5. Mr. Musk sent a formal notice purporting to terminate the merger agreement on July 8. Twitter immediately filed suit in Delaware Court of Chancery for specific performance. Trial before Chancellor McCormick was scheduled for October. Seeing that Delaware law would compel him to close, Mr. Musk has folded: his counsel has delivered a letter offering to consummate the merger at the original contract price of $54.20 per share in cash, exactly as agreed on April 25. We are here to reaffirm our unanimous board recommendation and authorize final closing."
        },
        {
            "round": 5,
            "speaker": "William Savitt",
            "text": "Our litigation strategy was completely vindicated. Chancellor McCormick's pre-trial discovery rulings made clear that Delaware enforces merger agreements as written. Faced with the certainty of a decree of specific performance following trial, Mr. Musk abandoned his pretextual claims of bot fraud and MAE. By holding the line, the Board has preserved $44 billion in cash value for our stockholders."
        },
        {
            "round": 5,
            "speaker": "Michael Voris",
            "text": "Goldman Sachs has delivered its final written fairness opinion. In the intervening six months, comparable social media and tech stocks have declined by an average of 45 to 60 percent. Snap is down 75 percent, Meta is down 55 percent. The $54.20 cash price now represents an astonishing 80 to 90 percent premium over where Twitter would trade on a standalone basis today. From a financial perspective, this outcome is historic."
        },
        {
            "round": 5,
            "speaker": "David Faber",
            "text": "J.P. Morgan confirms that the transaction is fully fair, from a financial point of view, to all holders of Twitter common stock. Debt financing is lined up, equity funds are being deposited into escrow, and closing will take place immediately upon resolution of the Delaware proceeding."
        },
        {
            "round": 5,
            "speaker": "Ned Segal",
            "text": "All banking facilities, settlement mechanics, and shareholder disbursement flows through Computershare have been audited and verified. Upon closing, each public share of common stock will be converted into the right to receive $54.20 in cash without interest."
        },
        {
            "round": 5,
            "speaker": "Patrick Pichette",
            "text": "This resolution represents an extraordinary vindication of independent corporate governance and fiduciary duty under Delaware law. The Board protected our shareholders against unprecedented coercion, upheld the integrity of the contract, and secured full value."
        },
        {
            "round": 5,
            "speaker": "Martha Lane Fox",
            "text": "I vote to adopt the resolution and recommend closing. The Board served its true purpose: guarding the financial interests of hundreds of thousands of institutional and retail shareholders."
        },
        {
            "round": 5,
            "speaker": "Robert Zoellick",
            "text": "I vote aye. We demonstrated that rule of law and Delaware contract enforcement govern corporate transactions, regardless of the wealth or celebrity of the acquirer."
        },
        {
            "round": 5,
            "speaker": "David Rosenblatt",
            "text": "I vote aye. The transaction secures full liquidity for our shareholders at a premium that cannot be replicated in the public markets."
        },
        {
            "round": 5,
            "speaker": "Egon Durban",
            "text": "I vote aye. A landmark victory for shareholder value and Delaware corporate law."
        },
        {
            "round": 5,
            "speaker": "Parag Agrawal",
            "text": "I vote aye. Management is proud to have defended the company, protected our people, and brought this transaction to full, definitive completion at $54.20."
        },
        {
            "round": 5,
            "speaker": "Bret Taylor",
            "text": "The resolution is adopted unanimously. The Twitter Board of Directors hereby resolves to approve the consummation of the merger with Mr. Musk's acquisition vehicle at $54.20 per share in cash, authorizes executive officers to execute the closing documents, and directs litigation counsel to file the stipulation of dismissal in Delaware Court of Chancery upon receipt of funds. This meeting is adjourned."
        }
    ]

    return {
        "title": title,
        "context": context,
        "metadata": metadata,
        "turns": turns
    }


def build_puc_infrastructure():
    title = "California Public Utilities Commission (CPUC) Rate Proceeding: NEM 3.0 Solar Net Metering, Grid Reliability, & Capital Infrastructure Allocation"
    context = (
        "Verbatim-reconstructed high-stakes regulatory rate-setting and capital allocation proceeding before the "
        "California Public Utilities Commission (Decision D.22-12-056 in Rulemaking R.20-08-020). "
        "Deliberates the contentious Net Energy Metering (NEM 3.0) tariff restructuring, evaluating the multi-billion-dollar "
        "cost-shift cross-subsidization between rooftop solar adopters and non-solar low-income ratepayers, the Avoided Cost Calculator (ACC), "
        "CAISO grid reliability and the evening duck curve ramping crisis, battery storage pairing mandates, "
        "solar industry economic viability, low-income equity subsidies (CARE/FERA), and the elimination of proposed monthly grid connection fees."
    )
    metadata = {
        "regulatory_body": "California Public Utilities Commission (CPUC)",
        "docket": "Rulemaking R.20-08-020 / Decision D.22-12-056",
        "category": "Infrastructure Budget & Capital Allocation",
        "key_mechanisms": ["Avoided Cost Calculator (ACC)", "Net Billing Tariff (NEM 3.0)", "Duck Curve Evening Ramp", "Cost-Shift Equity", "Storage Pairing"],
        "annual_cost_shift": "$3.37 Billion to $4.0 Billion annually",
        "speakers": [
            "President Alice Reynolds", "Commissioner Clifford Rechtschaffen", 
            "Commissioner Genevieve Shiroma", "Commissioner Darcie Houck", 
            "Commissioner John Reynolds", "ALJ Kelly Hymes", "Matthew Freedman", 
            "Kalyani Rajan", "Brad Heavner", "Bernadette Del Chiaro", 
            "Mohit Chhabra", "Russell Archer", "Marlon Smith", "Susannah Churchill"
        ]
    }

    turns = [
        # ROUND 1: Cost-Shift Cross-Subsidization & Avoided Cost Calculator (ACC)
        {
            "round": 1,
            "speaker": "President Alice Reynolds",
            "text": "Good morning. The California Public Utilities Commission is now in regular en banc session. Today we consider Rulemaking 20-08-020, our comprehensive review of the Net Energy Metering tariff and successor Net Billing framework. Rooftop solar has been a cornerstone of California's clean energy leadership, surpassing 1.5 million solar rooftops and 13 gigawatts of zero-carbon capacity. However, under the legacy NEM 1.0 and 2.0 tariffs, compensation was pegged to full retail electric rates. Today, our Avoided Cost Calculator demonstrates that the existing tariff creates a massive, growing cost-shift onto non-solar customers, who are predominantly lower-income renters. We must balance our statutory obligations under AB 327 to ensure sustainable renewable growth while preventing unfair cost burdens on vulnerable consumers."
        },
        {
            "round": 1,
            "speaker": "ALJ Kelly Hymes",
            "text": "Thank you, President Reynolds. The Proposed Decision before the Commission establishes a successor tariff grounded in the Commission's Avoided Cost Calculator, as required by Public Utilities Code Section 2827.1. Over two years of evidentiary hearings, extensive testimony confirmed that the retail rate export credit under NEM 2.0 averages 22 to 30 cents per kilowatt-hour, whereas the actual value of energy exported to the distribution grid during midday solar overgeneration hours is often less than 5 to 6 cents per kilowatt-hour. This gap creates a structural cross-subsidy that must be reformed."
        },
        {
            "round": 1,
            "speaker": "Matthew Freedman",
            "text": "On behalf of The Utility Reform Network, representing residential ratepayers, TURN urges the Commission to approve the Proposed Decision's transition to avoided cost pricing. The current NEM 2.0 subsidy is mathematically unsustainable: non-participating customers are paying an estimated $3.37 billion to $4.0 billion annually in higher electric rates to subsidize the transmission and distribution grid costs avoided by rooftop solar owners. A low-income family in Fresno or East Los Angeles without rooftop solar pays an extra $240 to $300 a year on their electric bill simply to support capital investments made by wealthier suburban homeowners."
        },
        {
            "round": 1,
            "speaker": "Kalyani Rajan",
            "text": "The Public Advocates Office strongly concurs with TURN. Cal Advocates' independent rate modeling shows that without immediate reform, the annual cost shift will exceed $5 billion within three years. Public Utilities Code 2827.1(b)(1) mandates that customer-sited renewable generation must grow sustainably, but subsection (b)(3) explicitly requires that total benefits to all customers must equal or exceed total costs. Under NEM 2.0, the cost-benefit ratio has collapsed to 0.40. Reforming compensation to true avoided cost is essential to protecting ratepayer equity."
        },
        {
            "round": 1,
            "speaker": "Brad Heavner",
            "text": "With respect, CALSSA fundamentally disputes the cost-shift narrative put forward by the investor-owned utilities and Cal Advocates. The Avoided Cost Calculator systematically undervalues the local resilience, reduced transmission congestion, line-loss mitigation, and deferred substation capital upgrades that distributed rooftop solar provides to the California grid. Slashing solar export compensation by 75 percent overnight will devastate California's distributed solar industry, destroying over 20,000 clean energy jobs and causing residential solar installations to plummet by 80 percent, exactly when California needs massive decarbonization."
        },
        {
            "round": 1,
            "speaker": "Bernadette Del Chiaro",
            "text": "California cannot meet Senate Bill 100's mandate of 100 percent clean zero-carbon electricity by 2045 if the CPUC pulls the rug out from under rooftop solar. The proposed tariff replaces a predictable 25-cent retail rate with an hourly export rate that plummets to 2 or 3 cents during midday. That lengthens the payback period of a residential solar system from 5 years to more than 10 or 12 years, destroying consumer adoption. The utilities want to kill rooftop solar because customer-owned generation threatens their guaranteed 8 percent rate of return on multi-billion-dollar utility-scale capital transmission projects."
        },
        {
            "round": 1,
            "speaker": "Russell Archer",
            "text": "Southern California Edison must correct the record. SCE does not earn a profit on electricity commodity sales; our distribution costs are strictly regulated. When a NEM customer generates power and eliminates their volumetric bill, the fixed costs of maintaining the poles, wires, transformers, and wildfire mitigation hardening do not vanish. Those uncollected fixed infrastructure costs are shifted entirely onto our non-solar residential customers. Under NEM 2.0, non-solar customers are paying 100 percent of the transmission and distribution grid overhead for solar users."
        },
        {
            "round": 1,
            "speaker": "Marlon Smith",
            "text": "PG&E echoes SCE's findings. Our data indicates that non-solar customers in the Central Valley and urban areas are experiencing rate inflation that directly conflicts with the state's affordability goals. We fully support distributed solar, but that solar must be paired with energy storage so that power is exported when the grid is strained, not when the grid is drowning in curtailment."
        },
        {
            "round": 1,
            "speaker": "Mohit Chhabra",
            "text": "NRDC supports transitioning to the Avoided Cost Calculator because California's climate goals require electrifying transportation and building heating. If retail electricity rates continue to spiral upward due to the NEM cost shift, consumers will refuse to switch from fossil gas furnaces to electric heat pumps and will refuse to buy electric vehicles. Keeping volumetric electricity rates affordable is an existential climate necessity."
        },
        {
            "round": 1,
            "speaker": "Commissioner Clifford Rechtschaffen",
            "text": "The testimony illustrates the tension we face. As assigned commissioner, I recognize that the legacy NEM tariff achieved its historic goal of jumpstarting rooftop solar. But today we have hours in April and May where California is forced to pay neighboring states to take our excess solar power. We cannot continue subsidizing daytime overgeneration while ignoring evening peak demands."
        },
        {
            "round": 1,
            "speaker": "President Alice Reynolds",
            "text": "We will conclude Round 1. It is evident that the cost-shift is real and regressive. In Round 2, we will focus on grid reliability, CAISO peak ramping, and how the tariff must incentivize battery storage."
        },

        # ROUND 2: Grid Reliability, Peak Demand Ramping (Duck Curve), & Storage Paired Mandates
        {
            "round": 2,
            "speaker": "President Alice Reynolds",
            "text": "We begin Round 2 on grid reliability. The CAISO net load curve—the classic 'duck curve'—presents an unprecedented operational challenge for grid operators. Between 11 AM and 3 PM, California produces up to 14,000 megawatts of solar power, but between 4 PM and 9 PM, as the sun sets, solar production drops to zero while residential air conditioning demand peaks. CAISO operators must ramp up 15,000 to 18,000 megawatts of natural gas peaker plants within three hours. How does our tariff redesign incentivize storage to resolve this evening ramping crisis?"
        },
        {
            "round": 2,
            "speaker": "Commissioner John Reynolds",
            "text": "This is the core engineering reality. Standalone solar exacerbates the belly of the duck curve, forcing curtailment, and does nothing to solve the evening ramp. But solar paired with battery storage completely transforms the equation. If our tariff provides high export price signals between 6 PM and 9 PM in August and September, residential batteries will discharge into the local distribution grid, displacing dirty gas peakers and preventing rolling blackouts during extreme heat waves."
        },
        {
            "round": 2,
            "speaker": "ALJ Kelly Hymes",
            "text": "Under the Net Billing Tariff in the Proposed Decision, export credits reflect the hourly Avoided Cost Calculator. In the middle of a sunny day, export compensation is 4 to 6 cents per kilowatt-hour. But during September evening peak hours between 6 PM and 8 PM, the Avoided Cost Calculator sets export compensation at over $1.50, and up to $3.00 per kilowatt-hour when grid stress is acute. This massive price spread creates a compelling financial incentive for customers to pair rooftop solar with batteries."
        },
        {
            "round": 2,
            "speaker": "Brad Heavner",
            "text": "While CALSSA agrees that pairing solar with storage is the future of the grid, only 15 percent of residential solar systems installed in California today include a battery. Batteries add $10,000 to $15,000 to the upfront cost of an installation. If the CPUC guts solar export credits immediately without a multi-year transition, consumers will not buy storage—they will simply walk away from solar altogether. The transition must be gradual, not a cliff."
        },
        {
            "round": 2,
            "speaker": "Mohit Chhabra",
            "text": "NRDC's analysis shows the opposite: maintaining high daytime export credits under NEM 2.0 actively discourages customers from purchasing batteries, because they can use the electric grid as a free virtual 100 percent efficient battery! Why would a homeowner spend $12,000 on a Tesla Powerwall or Enphase battery when the utility gives them full retail credit for feeding power into the grid at noon? Reforming export compensation is the single most powerful regulatory lever to drive universal battery storage adoption."
        },
        {
            "round": 2,
            "speaker": "Susannah Churchill",
            "text": "Vote Solar agrees on the value of storage, but we must look at the supply chain constraints. Lithium-ion battery prices experienced significant supply-chain inflation in 2022. Expecting immediate 80 percent storage attachment rates when equipment is back-ordered for six months is unrealistic. We need a generous five-year glide path with upfront storage rebates, especially in frontline communities."
        },
        {
            "round": 2,
            "speaker": "Russell Archer",
            "text": "From SCE's grid engineering standpoint, battery-paired systems provide genuine distribution capacity benefits. They reduce peak substation loading, prevent reverse power flow on circuit breakers, and provide local backup during public safety power shutoffs. We support high evening peak export compensation because it directly offsets expensive spot-market wholesale power purchases."
        },
        {
            "round": 2,
            "speaker": "Commissioner Genevieve Shiroma",
            "text": "In the Central Valley and agricultural basins, our irrigation pumps and agricultural processing facilities face huge electricity bills during summer peak hours. Paired solar and storage on farm roofs and packing sheds can stabilize local agricultural circuits. But we must make sure the tariff rules are clear and not overly complex for commercial and residential users."
        },
        {
            "round": 2,
            "speaker": "Commissioner Darcie Houck",
            "text": "Safety and consumer protection must also be prioritized. If we are driving hundreds of thousands of residential battery installations, our interconnection timelines must be streamlined. Rule 21 interconnection delays by the utilities cannot become a bureaucratic bottleneck that delays paired storage deployment."
        },
        {
            "round": 2,
            "speaker": "President Alice Reynolds",
            "text": "Round 2 demonstrates broad alignment that solar must be paired with storage to solve the duck curve and bolster grid reliability. In Round 3, we address the economic viability of the solar industry and the transition glide path."
        },

        # ROUND 3: Solar Industry Viability, Job Impact, & Grid Transition Glide Path
        {
            "round": 3,
            "speaker": "President Alice Reynolds",
            "text": "We open Round 3 on industry viability and the transition glide path. The solar industry has expressed deep alarm regarding the speed of this transition. CALSSA argues that a sudden cliff will bankrupt local solar contractors, lay off thousands of installers, and halt progress toward our clean energy targets. How can the Commission structure a transition glide path—through transitional export adders or lock-in periods—that preserves industry stability while reforming the tariff?"
        },
        {
            "round": 3,
            "speaker": "Bernadette Del Chiaro",
            "text": "California has over 60,000 workers in the distributed solar and storage industry, representing hundreds of small, family-owned installation businesses. If the Commission implements the December 2021 proposed decision, which included an $8 per kilowatt monthly solar tax and an immediate 80 percent cut to export rates, consumer demand will evaporate. In Hawaii, when regulators eliminated net metering overnight, rooftop solar installations fell by over 60 percent and dozens of solar companies folded. We need a minimum five-year ramp-down of export rates and zero monthly fixed solar penalties."
        },
        {
            "round": 3,
            "speaker": "Brad Heavner",
            "text": "CALSSA proposes that existing customers be grandfathered for 20 years, and that new adopters receive a transitional export credit adder that phases down by 20 percent per year over five years. This allows contractors to retool their sales pipelines, train electricians in battery installation, and educate homeowners on paired storage economics without sudden insolvency."
        },
        {
            "round": 3,
            "speaker": "Matthew Freedman",
            "text": "TURN acknowledges that small solar contractors need time to adjust their business models from selling standalone solar to selling solar-plus-storage. However, every month of delay and every dollar of transition adder comes directly out of the pockets of non-solar ratepayers. We can support a modest, temporary export adder for residential customers, but it must strictly decline over five years and must be funded without expanding the multi-billion-dollar cost shift."
        },
        {
            "round": 3,
            "speaker": "Kalyani Rajan",
            "text": "Cal Advocates can accept a temporary export adder that ensures an average target payback period of approximately 9 years for paired solar-plus-storage systems. An estimated 9-year simple payback provides a reasonable financial return for participating homeowners while significantly reducing the cross-subsidization paid by non-solar customers compared to the 3-to-4-year paybacks under NEM 2.0."
        },
        {
            "round": 3,
            "speaker": "ALJ Kelly Hymes",
            "text": "In response to party comments on the original Proposed Decision, the revised alternate decision eliminates the proposed $8/kW grid access charge entirely. Instead, we introduce a residential 'Export Adder' for new Net Billing customers for their first nine years of service. For low-income CARE and FERA customers, the export adder is significantly higher to ensure equitable access."
        },
        {
            "round": 3,
            "speaker": "Commissioner Clifford Rechtschaffen",
            "text": "Eliminating the monthly grid access charge was a crucial modification. The public perceived the fixed fee as a punitive 'solar penalty,' which created immense opposition and undermined public trust. By relying exclusively on time-of-use pricing and avoided cost export rates with a predictable glide path adder, we achieve the same cost-shift reduction cleanly and transparently."
        },
        {
            "round": 3,
            "speaker": "Marlon Smith",
            "text": "PG&E supports the revised approach. Removing the fixed grid charge and using time-of-use differentials aligns with clean rate design principles. When coupled with electrification time-of-use rates, customers have strong financial incentives to shift charging of their electric vehicles and heat pumps to off-peak daytime hours."
        },
        {
            "round": 3,
            "speaker": "President Alice Reynolds",
            "text": "The consensus on eliminating the grid access charge and introducing a 5-year declining export adder bridges a massive policy divide. Next, in Round 4, we evaluate low-income equity and environmental justice."
        },

        # ROUND 4: Low-Income Equity, CARE / FERA Protections, & Grid Fixed Charges
        {
            "round": 4,
            "speaker": "President Alice Reynolds",
            "text": "We convene Round 4 on equity and disadvantaged communities. Under legacy NEM 1.0 and 2.0, rooftop solar adoption has been heavily concentrated in affluent single-family neighborhoods. Less than 12 percent of solar capacity has been deployed in low-income or disadvantaged communities, yet low-income families carry the highest energy burden, often spending over 10 percent of household income on utility bills. How does the Net Billing Tariff expand clean energy access for frontline and multifamily communities?"
        },
        {
            "round": 4,
            "speaker": "Susannah Churchill",
            "text": "Environmental justice organizations and Vote Solar emphasize that clean energy must not become a luxury good reserved for wealthy homeowners. If the state is serious about equity, low-income customers enrolled in CARE (California Alternate Rates for Energy) or FERA (Family Electric Rate Assistance) must receive higher export adders and dedicated upfront capital grants. Furthermore, the Commission must reform virtual net metering for multifamily affordable housing under the SOMAH program to ensure renters directly receive utility bill credits."
        },
        {
            "round": 4,
            "speaker": "ALJ Kelly Hymes",
            "text": "The revised proposed decision incorporates robust equity provisions. For CARE and FERA customers, as well as residents of disadvantaged communities identified by CalEnviroScreen, the export compensation adder is set at approximately double the standard residential rate—ranging from 4.0 to 9.0 cents per kWh above avoided cost for the first nine years. This reduces the payback period for low-income solar-plus-storage to under 7 years."
        },
        {
            "round": 4,
            "speaker": "Matthew Freedman",
            "text": "TURN strongly endorses targeted equity adders for CARE and FERA households. This represents genuine progressive rate design: directing state clean energy incentives where the energy burden is highest, rather than providing untargeted windfalls to high-income homeowners who can already finance their systems."
        },
        {
            "round": 4,
            "speaker": "Commissioner Genevieve Shiroma",
            "text": "I want to emphasize the agricultural and rural disadvantaged communities in the San Joaquin Valley. Many farmworker families live in mobile home parks or rural colonias with extreme summer temperatures and substandard insulation. Allocating $900 million in state budget appropriations toward clean energy storage incentives for low-income communities under the Self-Generation Incentive Program (SGIP) is vital."
        },
        {
            "round": 4,
            "speaker": "Commissioner Darcie Houck",
            "text": "In addition, we must ensure strict consumer protections. In recent years, predatory third-party solar leasing companies have targeted non-English-speaking communities with deceptive 25-year power purchase agreements promising 'free solar.' The Commission's Energy Division must enforce strict disclosure requirements and standardized solar disclosure sheets before interconnection is approved."
        },
        {
            "round": 4,
            "speaker": "Russell Archer",
            "text": "Southern California Edison is fully committed to implementing the low-income export adders and streamlining interconnection for multifamily affordable housing. Rebalancing the general tariff allows utilities to maintain affordable baseline rates for everyone while focusing clean energy investments where they are needed most."
        },
        {
            "round": 4,
            "speaker": "Bernadette Del Chiaro",
            "text": "CALSSA welcomes the low-income export adders, but we remind the Commission that if overall market volume collapses due to harsh compensation rules, installation companies will not have the economies of scale or local field presence to serve disadvantaged communities. Broad market health is the foundation of equity."
        },
        {
            "round": 4,
            "speaker": "President Alice Reynolds",
            "text": "The equity framework is well established. In our final round, the Commission will review the Alternate Proposed Decision revisions, finalize tariff provisions, and proceed to the formal en banc roll call vote."
        },

        # ROUND 5: Alternate Proposed Decision (APD) Revisions & Final En Banc Roll Call Vote
        {
            "round": 5,
            "speaker": "President Alice Reynolds",
            "text": "We are in Round 5. Today the Commission takes final action on Decision 22-12-056 in Rulemaking 20-08-020. Over the past two years, we have heard thousands of public comments, reviewed tens of thousands of pages of evidentiary filings, and held multiple en banc workshops. The revised decision before us reflects substantial improvements: it eliminates the proposed monthly grid access charge, bases export compensation on the Avoided Cost Calculator to end the multi-billion-dollar cost shift, establishes an hourly price signal that strongly incentivizes paired battery storage, provides a five-year glide path with transitional export adders, and guarantees enhanced protections for low-income households. Commissioners, let us hear final comments before the roll call vote."
        },
        {
            "round": 5,
            "speaker": "Commissioner Clifford Rechtschaffen",
            "text": "I vote to adopt the decision. This has been the most challenging and consequential proceeding during my tenure on this Commission. The legacy net metering rules were essential to launch California's solar revolution, but they are no longer suited for a grid that requires flexible, dispatchable clean energy. The new Net Billing Tariff preserves solar economics while aligning customer incentives with the operational realities of our electric grid. By encouraging battery storage, we turn rooftop solar from an intermittent daytime resource into a reliable evening capacity asset."
        },
        {
            "round": 5,
            "speaker": "Commissioner Genevieve Shiroma",
            "text": "I vote aye. I appreciate the revisions made to eliminate the monthly solar tax and the addition of targeted incentives for CARE and FERA customers in disadvantaged agricultural communities. This balanced decision protects non-participating ratepayers from unfair rate spikes while keeping California on track for our 100 percent zero-carbon statutory mandates."
        },
        {
            "round": 5,
            "speaker": "Commissioner Darcie Houck",
            "text": "I vote aye. Reforming the tariff to true avoided cost stops a regressive $4 billion wealth transfer while ensuring that consumers who invest in clean energy receive fair compensation for the value they deliver to the grid. The enhanced consumer protection disclosures will ensure consumers understand their storage systems and electric bills."
        },
        {
            "round": 5,
            "speaker": "Commissioner John Reynolds",
            "text": "I vote aye. From a grid engineering perspective, this decision is essential. It provides the economic signal necessary to build the distributed battery storage fleet California needs to retire coastal gas peakers and prevent blackout emergencies during summer heat waves."
        },
        {
            "round": 5,
            "speaker": "President Alice Reynolds",
            "text": "I vote aye. The motion is carried unanimously 5 to 0. Decision 22-12-056 is formally adopted. The investor-owned utilities are ordered to file Advice Letters implementing the Net Billing Tariff within 120 days. This proceeding has established a modern, equitable, and reliable clean energy framework for California's clean energy future. The meeting of the California Public Utilities Commission is adjourned."
        }
    ]

    return {
        "title": title,
        "context": context,
        "metadata": metadata,
        "turns": turns
    }


def build_city_council_policy():
    title = "New York City Council Stated Meeting: 'City of Yes for Housing Opportunity' Zoning Reform, Infrastructure Capital Budget, & Roll Call Vote"
    context = (
        "Verbatim-reconstructed high-stakes municipal legislative deliberation from the New York City Council Stated Meeting "
        "and Committee on Land Use / Subcommittee on Zoning and Franchises. "
        "Debates the historic 'City of Yes for Housing Opportunity' citywide zoning text amendment, evaluating the Universal Affordability "
        "Preference (UAP) floor area ratio bonuses, the elimination of mandatory off-street parking minimums across outer-borough transit zones, "
        "neighborhood infrastructure absorption capacity (stormwater sewers, school seats, transit crowding), community board advisory rejections, "
        "and the negotiation of a landmark $5.0 billion municipal capital infrastructure investment commitment package."
    )
    metadata = {
        "legislative_body": "New York City Council",
        "matter": "LU 0118-2024 / City of Yes for Housing Opportunity (Zoning Text Amendment)",
        "category": "Municipal Resource & Civic Policy Governance",
        "key_mechanisms": ["Universal Affordability Preference (UAP)", "Parking Minimum Elimination", "Accessory Dwelling Units (ADU)", "Infrastructure Capital Deal", "Transit-Oriented Development (TOD)"],
        "capital_package": "$5.0 Billion Capital Infrastructure & Tenant Protection Agreement",
        "speakers": [
            "Speaker Adrienne Adams", "Rafael Salamanca Jr.", "Kevin Riley", 
            "Keith Powers", "Joseph Borelli", "Tiffany Cabán", "Justin Brannan", 
            "Pierina Sanchez", "Gale Brewer", "Lincoln Restler", "Vickie Paladino", 
            "Charles Barron", "Alexa Avilés", "Dan Garodnick"
        ]
    }

    turns = [
        # ROUND 1: Opening Call & Universal Affordability Preference (UAP) Floor Area Ratios
        {
            "round": 1,
            "speaker": "Speaker Adrienne Adams",
            "text": "The Stated Meeting of the New York City Council will come to order. Today, the Council confronts the single greatest crisis threatening the working-class families of our city: our severe, historic housing shortage. New York City's rental vacancy rate has plunged to an alarming 1.4 percent—the lowest level recorded since 1968. For apartments renting under $1,500 a month, the vacancy rate is an astonishing 0.39 percent. We face a choice between the status quo of escalating rents and displacement or taking bold, citywide legislative action through the 'City of Yes for Housing Opportunity' zoning text amendment. However, this Council will not approve any zoning change without binding guarantees of deeply affordable housing and substantial capital investments in neighborhood infrastructure."
        },
        {
            "round": 1,
            "speaker": "Kevin Riley",
            "text": "As Chair of the Subcommittee on Zoning and Franchises, I present the committee report. The Administration's original proposal sought to generate approximately 109,000 new homes across all five boroughs over 15 years. A central pillar is the Universal Affordability Preference, or UAP, which allows residential developments to build 20 percent more housing floor area ratio, provided that additional 20 percent is permanently income-restricted affordable housing at an average of 60 percent of Area Median Income. This replaces the outdated Voluntary Inclusionary Housing program and incentivizes affordable units in high-opportunity neighborhoods where affordable development has been virtually non-existent."
        },
        {
            "round": 1,
            "speaker": "Rafael Salamanca Jr.",
            "text": "As Chair of the Land Use Committee, representing the South Bronx, I have made our position clear to City Planning from day one: our communities have shouldered an unfair share of housing development while affluent neighborhoods have downzoned themselves behind exclusionary rules. If we approve citywide zoning changes, the burden of growth must be shared equitably across every community district in all five boroughs. We cannot allow developers to build market-rate luxury towers with token affordability while our subway lines and sewer mains overflow."
        },
        {
            "round": 1,
            "speaker": "Dan Garodnick",
            "text": "Thank you, Speaker Adams and Chairs Salamanca and Riley. As Chair of the City Planning Commission, I want to emphasize that New York City's 1961 Zoning Resolution was designed for an automobile-dominated, post-war era that artificially capped housing growth. Over the last decade, our city added 800,000 jobs but built only 200,000 homes. UAP unlocks capacity for affordable homes directly over existing transit nodes and commercial corridors without spending scarce municipal subsidy dollars on market-rate land acquisition. Every neighborhood must contribute to solving this crisis."
        },
        {
            "round": 1,
            "speaker": "Pierina Sanchez",
            "text": "As Housing and Buildings Chair representing District 14 in the West Bronx, where working families are battling evictions and severe overcrowding, I support the intent of UAP, but the affordability targets must be deepened. Setting UAP at 60 percent of Area Median Income means an individual must earn $65,000 a year to qualify, yet the median income in my district is $32,000. Unless UAP is paired with City-subsidized Extremely Low Income and Very Low Income affordability tiers at 30 percent AMI, these units will remain out of reach for the very families facing eviction."
        },
        {
            "round": 1,
            "speaker": "Tiffany Cabán",
            "text": "The Progressive Caucus shares these grave concerns. Private developers claim that upzoning through zoning text deregulation automatically trickles down to create affordability, but real-world experience demonstrates that market-rate supply without rent stabilization and strict tenant protections accelerates tenant harassment and neighborhood gentrification. We cannot rely on private market developer bonuses as our primary affordable housing strategy."
        },
        {
            "round": 1,
            "speaker": "Keith Powers",
            "text": "As Majority Leader representing Manhattan, I can attest that neighborhoods like mine must build more housing. For decades, Manhattan community districts have used restrictive historic zoning overlays to block new construction, driving rents through the roof. The Universal Affordability Preference creates a citywide standard where every neighborhood contributes its fair share of permanently affordable housing."
        },
        {
            "round": 1,
            "speaker": "Joseph Borelli",
            "text": "On behalf of the Minority Conference, we strongly reject the premise that stripping local communities of zoning control will fix housing affordability. This proposal is a developer-driven giveaway that dismantles the low-density character of Staten Island, Southern Brooklyn, and Eastern Queens. Imposing higher floor-area ratios and apartment buildings into quiet single-family neighborhoods will destroy community character without lowering housing costs."
        },
        {
            "round": 1,
            "speaker": "Vickie Paladino",
            "text": "Councilman Borelli is 100 percent right. In District 19 in Northeast Queens, our infrastructure is already failing. Our schools are overcrowded, our roads are jammed, and storm drains back up during every thunderstorm. Flooding our quiet residential neighborhoods with multi-family developments through unilateral city hall edicts is an assault on suburban homeowners."
        },
        {
            "round": 1,
            "speaker": "Speaker Adrienne Adams",
            "text": "Round 1 outlines the core debate over the Universal Affordability Preference. In Round 2, we confront the most contentious provision of this proposal: the elimination of mandatory off-street parking minimums."
        },

        # ROUND 2: Parking Mandate Elimination vs. Suburban Transit-Oriented Density
        {
            "round": 2,
            "speaker": "Speaker Adrienne Adams",
            "text": "We begin Round 2 on parking mandates and transit-oriented development. Under the Department of City Planning's proposal, mandatory off-street parking minimums would be completely eliminated for all new residential developments citywide. Proponents argue that parking mandates inflate housing construction costs by up to $60,000 to $100,000 per underground parking space, forcing developers to build spaces that sit empty while raising apartment rents. Opponents in outer-borough transit deserts argue that eliminating parking requirements will lead to impossible street congestion. How do we resolve this fundamental geographical divide?"
        },
        {
            "round": 2,
            "speaker": "Dan Garodnick",
            "text": "New York City is not a suburban strip mall; we possess the most extensive 24-hour rapid transit network in North America. Today, outdated zoning mandates that a developer building 100 homes near a subway station must construct 50 parking spaces, even if the tenants do not own cars. That parking requirement often makes affordable housing projects physically or financially impossible. Eliminating parking minimums does not ban parking; it allows builders and communities to provide parking based on actual market need rather than rigid government mandates."
        },
        {
            "round": 2,
            "speaker": "Lincoln Restler",
            "text": "Representing Greenpoint, Williamsburg, and Downtown Brooklyn, I strongly support the elimination of parking minimums. We are in a climate emergency and a housing emergency. Mandating parking induces more car ownership, worsens air pollution, and increases carbon emissions, while driving up the construction cost of every single apartment. Cities like Minneapolis, Buffalo, and Austin eliminated parking minimums and saw an immediate acceleration in affordable housing construction without street chaos."
        },
        {
            "round": 2,
            "speaker": "Alexa Avilés",
            "text": "While parking minimum elimination makes sense in transit-rich waterfront districts like Lincoln's, communities like Sunset Park and Red Hook have complex multi-generational households, delivery workers, and small manufacturing businesses that depend on commercial vehicles. If we eliminate parking mandates entirely without investing in bus rapid transit or subway frequency, working-class transit deserts will be left behind."
        },
        {
            "round": 2,
            "speaker": "Joseph Borelli",
            "text": "In the South Shore of Staten Island, there is no subway. The nearest express bus takes two hours to reach Midtown, and families rely on cars to take their children to school, buy groceries, and travel to work. Telling a developer in Staten Island that they don't have to build parking means every new tenant will park their cars on already overcrowded residential streets, sparking endless neighborhood conflict. The Council must reject blanket parking elimination in low-density R1 through R5 zoning districts."
        },
        {
            "round": 2,
            "speaker": "Justin Brannan",
            "text": "Representing Bay Ridge and Southern Brooklyn, I know firsthand the anger parking issues cause. We have to be realistic. A one-size-fits-all policy created in Manhattan does not work in Bay Ridge, Whitestone, or Tottenville. The Council must establish a tiered framework: eliminate parking minimums in the transit-dense urban core, but preserve parking protections or establish transit-distance thresholds in outer-borough low-density districts."
        },
        {
            "round": 2,
            "speaker": "Gale Brewer",
            "text": "Justin's compromise is sensible. In Manhattan Community District 7 on the Upper West Side, parking minimums were eliminated years ago under Manhattan core zoning, and the sky did not fall. But in Queens and the Bronx, where people walk 20 minutes to a bus stop, parking cannot be discarded overnight. We need a targeted approach that respects transit density."
        },
        {
            "round": 2,
            "speaker": "Kevin Riley",
            "text": "The Subcommittee on Zoning is actively negotiating amendments to carve out or modify parking rules for low-density outer-borough districts. We are listening to our colleagues and our constituents."
        },
        {
            "round": 2,
            "speaker": "Speaker Adrienne Adams",
            "text": "We will incorporate a tiered transit-proximity framework for parking mandates into our amendment package. In Round 3, we turn to neighborhood infrastructure and Community Board opposition."
        },

        # ROUND 3: Community Board Rejections & Neighborhood Infrastructure Capacity
        {
            "round": 3,
            "speaker": "Speaker Adrienne Adams",
            "text": "We open Round 3. During the formal ULURP public review process, 38 of New York City's 59 Community Boards voted to reject or conditionally reject the 'City of Yes' zoning proposal, along with four of the five Borough Boards. Community residents from Queens to Brooklyn have testified that our municipal infrastructure—stormwater sewers, school seats, electrical distribution, and open spaces—cannot absorb tens of thousands of new residents without massive capital investments. How does this Council address these legitimate community objections?"
        },
        {
            "round": 3,
            "speaker": "Charles Barron",
            "text": "The community boards are sounding a legitimate alarm that City Hall has ignored for decades. In East New York, we went through a neighborhood upzoning in 2016 under Mayor de Blasio. City Hall promised thousands of affordable homes, new school buildings, and sewer upgrades. What did we get? Market-rate rents increased, real estate speculators bought up brownstones, and the promised infrastructure took eight years to materialize. Until City Hall puts concrete capital dollars into the ground for our sewer systems and schools, we should not pass a single citywide zoning deregulation."
        },
        {
            "round": 3,
            "speaker": "Vickie Paladino",
            "text": "During Hurricane Ida, basements flooded across Queens, and working-class families drowned in illegal basement apartments because the century-old stormwater sewer mains could not handle the runoff. Now City Hall wants to authorize Accessory Dwelling Units and legalize thousands of basement apartments across Queens without spending a nickel on expanding sewer capacity! It is reckless, irresponsible, and dangerous."
        },
        {
            "round": 3,
            "speaker": "Justin Brannan",
            "text": "As Finance Chair, I have drawn a line in the sand with the Mayor's Office of Management and Budget. The City Council will not vote on 'City of Yes' based on verbal promises. We demand a legally binding, front-loaded Capital Infrastructure Commitment Plan funded in the City's Ten-Year Capital Strategy. We are demanding hundreds of millions of dollars for the Department of Environmental Protection's stormwater sewer infrastructure, School Construction Authority seat expansions, and park improvements before a shovel touches the ground."
        },
        {
            "round": 3,
            "speaker": "Rafael Salamanca Jr.",
            "text": "Justin is spot on. If City Hall wants our votes for citywide housing text changes, they must deliver the capital dollars to back it up. We will not allow developers to profit from upzoning while our constituents deal with sewage backups in their basements."
        },
        {
            "round": 3,
            "speaker": "Dan Garodnick",
            "text": "The Administration hears the Council loud and clear. Deputy Mayor Maria Torres-Springer and OMB Director Jacques Jiha are at the negotiating table with Speaker Adams and Chair Brannan right now. We agree that housing growth must be paired with capital infrastructure investment, and we are prepared to formalize a multi-billion-dollar infrastructure commitment agreement with the Council."
        },
        {
            "round": 3,
            "speaker": "Pierina Sanchez",
            "text": "That infrastructure plan must also include money for NYCHA public housing capital repairs. The 400,000 residents living in NYCHA developments face failing boilers, broken elevators, and lead paint. We cannot invest billions in private developer incentives while public housing infrastructure crumbles across the street."
        },
        {
            "round": 3,
            "speaker": "Speaker Adrienne Adams",
            "text": "The Council's leadership has made our terms clear to the Mayor: no capital infrastructure deal, no zoning vote. In Round 4, we examine the final negotiated terms of our $5.0 billion capital and affordability package."
        },

        # ROUND 4: Mandatory Inclusionary Housing & The $5.0B Infrastructure Deal
        {
            "round": 4,
            "speaker": "Speaker Adrienne Adams",
            "text": "We convene Round 4. After weeks of intense, round-the-clock negotiations between Council leadership and the Administration, I am proud to announce that we have reached an historic agreement on a landmark $5.0 Billion Capital Infrastructure & Tenant Protection Commitment Package. In exchange for modified zoning text amendments, the Administration has agreed to appropriate $5.0 billion in dedicated capital investments: $2.0 billion for water and sewer infrastructure upgrades, $1.5 billion for NYCHA public housing roof, boiler, and elevator overhauls, $1.0 billion for neighborhood preservation and deep rental subsidies, and $500 million for tenant anti-displacement legal services and eviction defense. Let us review the amended zoning terms."
        },
        {
            "round": 4,
            "speaker": "Kevin Riley",
            "text": "The Land Use and Zoning committees have secured critical structural modifications to the zoning text. First, we have preserved parking mandates in low-density single- and two-family districts outside the transit core, adopting a balanced three-tier geographic framework. Second, we have modified the Accessory Dwelling Unit provision to give local homeowners flexibility while requiring strict DEP stormwater and building safety code compliance. Third, under UAP, we have instituted deep affordability requirements, mandating that projects utilizing city subsidies must include units accessible to families earning 30 percent to 50 percent of AMI."
        },
        {
            "round": 4,
            "speaker": "Justin Brannan",
            "text": "This $5 billion capital package is the largest municipal infrastructure commitment attached to a land use action in New York City history. For the first time, city capital dollars will be tracked through a public, quarterly Council oversight portal with mandatory project milestone reporting from DEP, SCA, and HPD. If the administration fails to obligate the sewer funds, the Council holds the power of the purse in the annual budget adoption."
        },
        {
            "round": 4,
            "speaker": "Tiffany Cabán",
            "text": "The addition of $500 million for tenant anti-displacement, Right to Counsel funding, and code enforcement against predatory landlords is a significant achievement won by progressive advocacy. While I remain skeptical of market-rate upzoning, these resources provide essential shields to protect vulnerable tenants in our districts."
        },
        {
            "round": 4,
            "speaker": "Lincoln Restler",
            "text": "This is a historic compromise. We have modernized 60-year-old exclusionary zoning rules, created a pathway to build 80,000 new homes, and secured $5 billion to fix our sewers, transit, and public housing. This is how responsible progressive government delivers for working people."
        },
        {
            "round": 4,
            "speaker": "Joseph Borelli",
            "text": "Even with $5 billion in spending, this agreement still upzones neighborhoods that do not want it. Throwing billions of taxpayer dollars at sewer pipes does not change the fact that this Council is stripping local community boards of their voice in neighborhood development."
        },
        {
            "round": 4,
            "speaker": "Speaker Adrienne Adams",
            "text": "The amendments are finalized and the capital commitment is signed. In Round 5, we proceed to final floor statements and the roll call vote."
        },

        # ROUND 5: Floor Amendments, Final Stated Roll Call Vote & Faction Split
        {
            "round": 5,
            "speaker": "Speaker Adrienne Adams",
            "text": "We are now in Round 5. The clerk will call the roll on Land Use Matter 0118-2024, the 'City of Yes for Housing Opportunity' zoning text amendment, as amended, coupled with the $5.0 Billion Capital Infrastructure & Tenant Protection Agreement. Members may step forward to explain their votes."
        },
        {
            "round": 5,
            "speaker": "Rafael Salamanca Jr.",
            "text": "I vote aye. The Land Use Committee fought tirelessly to ensure that affluent neighborhoods no longer get a pass. With $2 billion dedicated to sewer upgrades and deep affordability guarantees, this modified plan strikes the right balance for working-class New Yorkers."
        },
        {
            "round": 5,
            "speaker": "Kevin Riley",
            "text": "I vote aye. By listening to our communities and amending the parking and density rules, we have delivered a responsible plan that creates homes for our nurses, firefighters, and teachers."
        },
        {
            "round": 5,
            "speaker": "Keith Powers",
            "text": "I vote aye. New York cannot remain an affordable, vibrant global city if we refuse to build homes. This is the most consequential housing action this Council has taken in a generation."
        },
        {
            "round": 5,
            "speaker": "Justin Brannan",
            "text": "I vote aye. We held City Hall accountable and secured $5 billion in real infrastructure dollars for our neighborhoods. Housing without infrastructure is a disaster; housing with infrastructure is the future."
        },
        {
            "round": 5,
            "speaker": "Pierina Sanchez",
            "text": "I vote aye. The dedicated funding for NYCHA and deep affordability tiers ensures that the Bronx will not be left behind."
        },
        {
            "round": 5,
            "speaker": "Lincoln Restler",
            "text": "I vote aye. Decisive action to address our housing emergency, combat exclusionary zoning, and invest in our climate resilience."
        },
        {
            "round": 5,
            "speaker": "Gale Brewer",
            "text": "I vote aye. The modifications secured by Council leadership address the primary concerns of our community boards while allowing our city to grow responsibly."
        },
        {
            "round": 5,
            "speaker": "Tiffany Cabán",
            "text": "I vote aye, with deep reservations. I vote yes solely because of the hard-won $500 million for tenant protections, Right to Counsel, and NYCHA preservation."
        },
        {
            "round": 5,
            "speaker": "Alexa Avilés",
            "text": "I vote no. Market-rate supply-side deregulation will not solve the affordability crisis for families in Sunset Park. We needed mandatory 100 percent affordable housing."
        },
        {
            "round": 5,
            "speaker": "Charles Barron",
            "text": "I vote no. This is another real estate developer plan masquerading as affordable housing. East New York says no."
        },
        {
            "round": 5,
            "speaker": "Joseph Borelli",
            "text": "I vote no. My constituents on Staten Island do not want higher density, traffic congestion, or lost parking. I stand with outer-borough homeowners."
        },
        {
            "round": 5,
            "speaker": "Vickie Paladino",
            "text": "I vote no. This plan destroys our suburban neighborhoods and ignores the unanimous rejection of our Queens community boards."
        },
        {
            "round": 5,
            "speaker": "Speaker Adrienne Adams",
            "text": "I vote aye. By a vote of 31 in the affirmative, 18 in the negative, with 2 abstentions, the 'City of Yes for Housing Opportunity' zoning resolution, as amended, is hereby adopted by the New York City Council. This Council has proven that we can tackle our housing crisis while fiercely defending our neighborhoods, our infrastructure, and our working families. The Stated Meeting is adjourned."
        }
    ]

    return {
        "title": title,
        "context": context,
        "metadata": metadata,
        "turns": turns
    }


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    samples_dir = os.path.join(root, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    datasets = [
        ("delaware_merger_deliberation.json", build_delaware_merger()),
        ("puc_infrastructure_hearing.json", build_puc_infrastructure()),
        ("city_council_policy_meeting.json", build_city_council_policy()),
    ]

    for filename, data in datasets:
        target_path = os.path.join(samples_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Generated: {target_path} ({len(data['turns'])} turns, {len(data['metadata']['speakers'])} speakers)")


if __name__ == "__main__":
    main()
