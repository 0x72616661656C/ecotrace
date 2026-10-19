import * as React from 'react';
import { cn } from '@/lib/utils';

const SidebarContext = React.createContext({
  expanded: true,
  setExpanded: () => {},
});

export function SidebarProvider({ children }) {
  const [expanded, setExpanded] = React.useState(true);
  return (
    <SidebarContext.Provider value={{ expanded, setExpanded }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebar() {
  return React.useContext(SidebarContext);
}

export function Sidebar({ className, variant = 'floating', children, ...props }) {
  const isFloating = variant === 'floating';

  return (
    <aside
      className={cn(
        'fixed top-4 left-4 bottom-4 z-40 flex w-88 flex-col transition-all duration-300 pointer-events-auto',
        isFloating
          ? 'rounded-2xl border border-sidebar-border bg-sidebar/95 text-sidebar-foreground backdrop-blur-xl shadow-2xl ring-1 ring-black/5 dark:ring-white/10'
          : 'border-r border-sidebar-border bg-sidebar text-sidebar-foreground',
        className
      )}
      {...props}
    >
      {children}
    </aside>
  );
}

export function SidebarHeader({ className, children, ...props }) {
  return (
    <div
      className={cn('flex flex-col gap-2 p-4 border-b border-sidebar-border', className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function SidebarContent({ className, children, ...props }) {
  return (
    <div
      className={cn(
        'flex flex-1 flex-col gap-4 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-sidebar-border',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function SidebarFooter({ className, children, ...props }) {
  return (
    <div
      className={cn('flex flex-col gap-2 p-4 border-t border-sidebar-border mt-auto', className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function SidebarGroup({ className, children, ...props }) {
  return (
    <div className={cn('flex flex-col gap-2', className)} {...props}>
      {children}
    </div>
  );
}

export function SidebarGroupLabel({ className, children, ...props }) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground px-1',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function SidebarGroupContent({ className, children, ...props }) {
  return <div className={cn('flex flex-col gap-2', className)} {...props} />;
}
