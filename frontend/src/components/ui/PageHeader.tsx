import React from "react";

export interface PageHeaderProps {
  title: string;
  description: string;
  badge?: string;
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  badge,
  actions,
  className = "",
}: PageHeaderProps) {
  return (
    <div
      className={`flex flex-col gap-3 border-b border-ink-200 pb-5 md:flex-row md:items-start md:justify-between ${className}`}
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2.5">
          <h1 className="text-xl font-bold tracking-tight text-ink-900 sm:text-[22px]">
            {title}
          </h1>
          {badge && (
            <span className="pt-chip border-ink-200 bg-ink-100 text-ink-600">
              {badge}
            </span>
          )}
        </div>
        <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-ink-500">
          {description}
        </p>
      </div>

      {actions && (
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      )}
    </div>
  );
}
