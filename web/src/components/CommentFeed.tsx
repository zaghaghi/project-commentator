import { useEffect, useRef, useState } from "react";
import type { Comment } from "../useComments";

// A single roast. The reasoning ("thinking") and the screenshot are both
// hidden by default — small buttons in the header reveal them on demand, so
// the feed reads as a clean stream of quips.
function CommentRow({ c }: { c: Comment }) {
  const [showShot, setShowShot] = useState(false);
  const [showThinking, setShowThinking] = useState(false);

  return (
    <div className="entry spy">
      <div className="who">
        <span>Commentator · {new Date(c.ts).toLocaleTimeString()}</span>
        {c.thinking && (
          <button
            type="button"
            className="shot-toggle"
            onClick={() => setShowThinking((s) => !s)}
          >
            {showThinking ? "hide thinking" : "view thinking"}
          </button>
        )}
        {c.thumb && (
          <button
            type="button"
            className="shot-toggle"
            onClick={() => setShowShot((s) => !s)}
          >
            {showShot ? "hide screenshot" : "view screenshot"}
          </button>
        )}
      </div>
      {showThinking && c.thinking && <div className="thinking">{c.thinking}</div>}
      <div className="body">{c.text}</div>
      {showShot && c.thumb && (
        <img
          className="thumb"
          src={`data:image/png;base64,${c.thumb}`}
          alt="screen capture"
        />
      )}
    </div>
  );
}

export function CommentFeed({ comments }: { comments: Comment[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [comments]);

  return (
    <div className="transcript">
      {comments.map((c) => (
        <CommentRow key={c.id} c={c} />
      ))}
      <div ref={endRef} />
    </div>
  );
}