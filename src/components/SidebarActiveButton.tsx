'use client';
import Link from "next/link";
import React from "react";
import { SidebarMenuButton } from "@/components/ui/sidebar"
import { usePathname } from "next/navigation";

interface MenuItem {
  title: string;
  url: string;
  children:React.ReactNode
}

export default function SidebarActiveButton({ url, title, children }: MenuItem) {
  const pathname = usePathname();

  const checkMatch = () => {
    if(title === "Talent finder" && pathname.includes("talent-finder")){
      return true
    } else if(url === pathname){
      return true
    }else{
      return false
    }
  }

  // console.log("checkMatch", checkMatch())
  // console.log("url === pathname", url === pathname)

  return (
    <SidebarMenuButton asChild isActive={checkMatch()}>
      <Link href={url}>
        {children}
        <span>{title}</span>
      </Link>
    </SidebarMenuButton>
  );
}

// 'use client';
// import Link from "next/link";
// import React from "react";
// import { LucideProps } from "lucide-react";
// import { SidebarMenuButton } from "@/components/ui/sidebar"
// import { usePathname } from "next/navigation";

// interface MenuItem {
//   title: string;
//   url: string;
//   icon: React.ElementType
// }

// interface SidebarActiveButtonProps {
//   item: MenuItem;
// }

// export default function SidebarActiveButton({ item }: SidebarActiveButtonProps) {
//   const pathname = usePathname();
//   const Icon = iconMapping[item.icon];


//   return (
//     <SidebarMenuButton asChild isActive={item.url === pathname}>
//       <Link href={item.url}>
//         <item.icon />
//         <span>{item.title}</span>
//       </Link>
//     </SidebarMenuButton>
//   );
// }