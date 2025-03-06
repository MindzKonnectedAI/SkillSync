import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Providers } from "@/lib/providers"

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
        <Providers>
            <SidebarProvider>
                <AppSidebar />
                <SidebarInset>
                    <div className="p-4">
                        {children}
                    </div>
                </SidebarInset>
            </SidebarProvider>
        </Providers>
    )
}
