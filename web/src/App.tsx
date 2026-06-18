import { useComments } from "./useComments";
import { CommentFeed } from "./components/CommentFeed";
import { StartupGate } from "./components/StartupGate";

export default function App() {
  const { status, comments } = useComments();

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          PROJECT COMMENTATOR <small>// it watches, it roasts</small>
        </div>
      </header>
      {!status || status.phase !== "ready" ? (
        <StartupGate status={status} />
      ) : (
        <CommentFeed comments={comments} />
      )}
    </div>
  );
}