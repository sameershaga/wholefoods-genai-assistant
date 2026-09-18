"use client";

import { FormEvent, useEffect, useState } from "react";

type DemoStore = { store_id: string; label: string };
type QueryResult = {
  request_id: string;
  store_id: string;
  answer: string;
  citations: string[];
  model: string;
};

const suggestions = ["Do we have oat milk?", "What was delivered?"];
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // The status-based message below is safe when the server did not return JSON.
  }
  return `The assistant returned an error (${response.status}).`;
}

export function AssistantConsole() {
  const [stores, setStores] = useState<DemoStore[]>([]);
  const [storeId, setStoreId] = useState("");
  const [question, setQuestion] = useState(suggestions[0]);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState("");
  const [loadingStores, setLoadingStores] = useState(true);
  const [asking, setAsking] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    async function loadStores() {
      try {
        const response = await fetch("/v1/demo/stores", {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        const options = (await response.json()) as DemoStore[];
        if (options.length === 0) throw new Error("No synthetic demo stores are configured.");
        setStores(options);
        setStoreId(options[0].store_id);
      } catch (reason) {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Unable to load synthetic stores.");
      } finally {
        if (!controller.signal.aborted) setLoadingStores(false);
      }
    }
    void loadStores();
    return () => controller.abort();
  }, []);

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = question.trim();
    if (!query || !storeId || asking) return;
    setAsking(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch("/v1/demo/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, store_id: storeId }),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      setResult((await response.json()) as QueryResult);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to reach the assistant.");
    } finally {
      setAsking(false);
    }
  }

  return (
    <section className="console" aria-labelledby="console-title">
      <div className="consoleIntro">
        <div><p className="eyebrow">Assistant console</p><h2 id="console-title">Ask within a synthetic store scope</h2></div>
        <span className="isolationBadge">Store isolation enabled</span>
      </div>
      <div className="queryGrid">
        <form className="queryPanel" onSubmit={ask}>
          <label htmlFor="store">Synthetic store</label>
          <select id="store" value={storeId} onChange={(event) => setStoreId(event.target.value)} disabled={loadingStores || asking || stores.length === 0}>
            {loadingStores && <option value="">Loading stores…</option>}
            {!loadingStores && stores.length === 0 && <option value="">Unavailable</option>}
            {stores.map((store) => <option value={store.store_id} key={store.store_id}>{store.label} ({store.store_id})</option>)}
          </select>
          <label htmlFor="question">Operational question</label>
          <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={2000} required disabled={asking} />
          <div className="suggestions" aria-label="Suggested questions">
            {suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => setQuestion(suggestion)} disabled={asking}>{suggestion}</button>)}
          </div>
          <button className="askButton" type="submit" disabled={asking || loadingStores || !storeId || !question.trim()}>
            <span>{asking ? "Retrieving store-scoped evidence…" : "Ask the assistant"}</span><span aria-hidden="true">→</span>
          </button>
          <p className="formNote">The API resolves this selection to a server-side demo identity; the browser cannot supply arbitrary identity metadata.</p>
        </form>

        <div className="answerPanel" aria-live="polite" aria-busy={asking}>
          {error && <div className="answerState errorState" role="alert"><span className="answerIcon" aria-hidden="true">!</span><h3>Something went wrong</h3><p>{error}</p></div>}
          {!error && asking && <div className="answerState"><span className="answerIcon loadingIcon" aria-hidden="true">✦</span><h3>Building a grounded answer</h3><p>Applying identity scope, retrieval, reranking, and generation.</p></div>}
          {!error && !asking && !result && <div className="answerState"><span className="answerIcon" aria-hidden="true">✦</span><h3>Your grounded answer will appear here</h3><p>The response will include its selected store, model, and source document IDs.</p></div>}
          {!error && !asking && result && <article className="answerResult">
            <div className="resultMeta"><span>Scoped to {result.store_id}</span><span>Model: {result.model}</span></div>
            <h3>Grounded answer</h3><p className="answerText">{result.answer}</p>
            <h4>Sources</h4>
            {result.citations.length > 0 ? <ul className="citations">{result.citations.map((citation) => <li key={citation}>{citation}</li>)}</ul> : <p className="noCitations">No source IDs were returned.</p>}
          </article>}
          <div className="trustRow"><span>Scoped identity</span><span>Retrieved evidence</span><span>Citations</span></div>
        </div>
      </div>
    </section>
  );
}
