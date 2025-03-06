'use client';
import Link from "next/link";
import React from "react";
import { SidebarMenuSubButton } from "@/components/ui/sidebar"
import { usePathname } from "next/navigation";
import { useParams } from "next/navigation";

interface MenuItem {
  title: string;
  url: string;
}

export default function SidebarMenuSubActiveButton({ url, title }: MenuItem) {
  const pathname = usePathname();
  const {slug} = useParams();

  // console.log("pathname", pathname)
  // console.log("id", slug)

  return (
    <SidebarMenuSubButton asChild isActive={url === pathname}>
      <Link href={url}>
        <span>{title}</span>
      </Link>
    </SidebarMenuSubButton>
  );
}
