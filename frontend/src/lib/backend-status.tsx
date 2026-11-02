"use client";

import React, { createContext, useContext } from "react";

export type BackendStatus = "connected" | "disconnected" | "checking";

/**
 * The workspace shell owns the single backend health request. Views that need to
 * show connectivity (the Dashboard) read it from here instead of issuing their
 * own request, so one page load never fans out to /health more than once.
 */
const BackendStatusContext = createContext<BackendStatus>("checking");

export function BackendStatusProvider({
  status,
  children,
}: {
  status: BackendStatus;
  children: React.ReactNode;
}) {
  return (
    <BackendStatusContext.Provider value={status}>
      {children}
    </BackendStatusContext.Provider>
  );
}

export function useBackendStatus(): BackendStatus {
  return useContext(BackendStatusContext);
}
