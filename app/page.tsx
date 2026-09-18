const pipeline = [
  "Query",
  "Identity / store scope",
  "Embedding",
  "Filtered retrieval",
  "Reranking",
  "Answer + citations",
];

const metrics = [
  ["Cases", "Bundled regression set"],
  ["Retrieval", "Store-scoped evidence"],
  ["Answers", "Deterministic local model"],
  ["Citations", "Source document IDs"],
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

      <section className="console" aria-labelledby="console-title">
        <div className="consoleIntro">
          <div>
            <p className="eyebrow">Assistant console</p>
            <h2 id="console-title">Ask within a synthetic store scope</h2>
          </div>
          <span className="isolationBadge">Store isolation enabled</span>
        </div>

        <div className="queryGrid">
          <form className="queryPanel">
            <label htmlFor="store">Synthetic store</label>
            <select id="store" name="store" defaultValue="BROOKLYN-01">
              <option value="BROOKLYN-01">Brooklyn 01</option>
              <option value="MANHATTAN-01">Manhattan 01</option>
              <option value="QUEENS-01">Queens 01</option>
            </select>
            <label htmlFor="question">Operational question</label>
            <textarea id="question" name="question" defaultValue="Do we have oat milk?" />
            <div className="suggestions" aria-label="Suggested questions">
              <button type="button">Do we have oat milk?</button>
              <button type="button">What was delivered?</button>
            </div>
            <button className="askButton" type="button">Ask the assistant <span>→</span></button>
            <p className="formNote">Live API interaction will be connected in the next implementation increment.</p>
          </form>

          <div className="answerPanel" aria-label="Answer preview">
            <div className="answerEmpty">
              <span className="answerIcon" aria-hidden="true">✦</span>
              <h3>Your grounded answer will appear here</h3>
              <p>The response will include its selected store, model, and source document IDs.</p>
            </div>
            <div className="trustRow">
              <span>Scoped identity</span><span>Retrieved evidence</span><span>Citations</span>
            </div>
          </div>
        </div>
      </section>

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
