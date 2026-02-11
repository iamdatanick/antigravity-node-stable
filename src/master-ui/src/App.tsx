import { Routes, Route, Navigate } from "react-router";
import AppShell from "./components/layout/AppShell";
import ErrorBoundary from "./components/ErrorBoundary";
import Dashboard from "./pages/Dashboard";
import Chat from "./pages/Chat";
import Unavailable from "./pages/Unavailable";
import Budget from "./pages/Budget";
import Services from "./pages/Services";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <ErrorBoundary fallbackTitle="Antigravity Node encountered an error">
      <AppShell>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route
            path="/logs"
            element={
              <Unavailable
                feature="Logs"
                reason="Log streaming requires OpenSearch, which was removed in v14.1 Phoenix. Logs are available via docker compose logs."
              />
            }
          />
          <Route
            path="/memory"
            element={
              <Unavailable
                feature="Memory"
                reason="Episodic memory requires StarRocks, which was removed in v14.1 Phoenix. Memory is stored in etcd."
              />
            }
          />
          <Route
            path="/query"
            element={
              <Unavailable
                feature="Query"
                reason="SQL query requires StarRocks, which was removed in v14.1 Phoenix."
              />
            }
          />
          <Route
            path="/workflows"
            element={
              <Unavailable
                feature="Workflows"
                reason="Workflow visualization requires Argo, which was replaced by AsyncDAGEngine in v14.1 Phoenix."
              />
            }
          />
          <Route path="/budget" element={<Budget />} />
          <Route path="/services" element={<Services />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </ErrorBoundary>
  );
}
