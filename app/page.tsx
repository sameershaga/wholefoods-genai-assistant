import { AssistantConsole } from "./assistant-console";
import bundledResults from "@/evaluation/bundled-results.json";

const pipeline = [
  "Query",
  "Identity / store scope",
  "Embedding",
  "Filtered retrieval",
  "Reranking",
  "Answer + citations",
];

const formatRate = (rate: number) => `${Math.round(rate * 100)}%`;

const metrics = [
  ["Cases", String(bundledResults.case_count)],
  ["Retrieval hit rate", formatRate(bundledResults.retrieval_hit_rate)],
  ["Correct-store retrieval", formatRate(bundledResults.correct_store_retrieval_rate)],
  ["Abstention success", formatRate(bundledResults.abstention_success_rate)],
  ["Answer correctness", formatRate(bundledResults.answer_correctness)],
  ["Citation correctness", formatRate(bundledResults.citation_correctness)],
];

export default function Home() {
  return (
    <main>
      <header className="siteHeader">
        <a className="brand" href="#top" aria-label="Store Operations AI home">
          <span className="brandMark" aria-hidden="true">SO</span>
          <span>Store Operations AI</span>
        </a>
        <span className="referenceLabel">Synthetic reference implementation</span>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">Enterprise AI engineering portfolio</p>
          <h1>Grounded answers for store operations.</h1>
          <p className="lede">
            Explore how identity-aware retrieval can answer operational questions from
            synthetic, store-scoped evidence—and show exactly which sources support the answer.
          </p>
        </div>
        <div className="scopeCard">
          <span className="statusDot" aria-hidden="true" />
          <div>
            <strong>Local deterministic demo</strong>
            <p>No paid model or cloud credentials required.</p>
          </div>
        </div>
      </section>

      <AssistantConsole />

      <section className="section" aria-labelledby="pipeline-title">
        <p className="eyebrow">How it works</p>
        <h2 id="pipeline-title">Security before similarity</h2>
        <p className="sectionLead">
          Store metadata filters retrieval before reranking, so candidates outside the trusted
          identity scope never become answer context. Retrieving broadly within that scope and
          reranking a smaller set balances recall with focused generation.
        </p>
        <ol className="pipeline">
          {pipeline.map((step, index) => (
            <li key={step}><span>{String(index + 1).padStart(2, "0")}</span>{step}</li>
          ))}
        </ol>
        <p className="citationNote">Citations make every answer inspectable instead of asking the user to trust an unsupported response.</p>
      </section>

      <section className="section evaluation" aria-labelledby="evaluation-title">
        <div>
          <p className="eyebrow">Engineering &amp; evaluation</p>
          <h2 id="evaluation-title">Built to be tested offline</h2>
          <p className="sectionLead">
            The bundled evaluation exercises retrieval, store isolation, abstention, answer
            correctness, and citations using a tiny deterministic synthetic regression set.
            These results are not evidence of real-world model accuracy or production performance.
          </p>
          <p className="citationNote">
            Checked-in reference results, verified against the bundled dataset by the backend test
            suite—not a live production evaluation.
          </p>
        </div>
        <dl className="metricGrid">
          {metrics.map(([term, detail]) => (
            <div key={term}><dt>{term}</dt><dd>{detail}</dd></div>
          ))}
        </dl>
      </section>

      <footer>
        <p><strong>Synthetic enterprise GenAI/RAG reference implementation.</strong></p>
        <p>Not an actual Whole Foods production application. No proprietary systems, data, metrics, or users are represented.</p>
      </footer>
    </main>
  );
}
