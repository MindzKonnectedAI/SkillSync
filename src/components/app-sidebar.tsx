'use client'
import React from "react"
import { CircleUser } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSubItem,
  SidebarGroupContent,
  SidebarFooter
} from "@/components/ui/sidebar"
import SidebarActiveButton from "./SidebarActiveButton";
import Image from "next/image";
import { NavUser } from "./nav-user";
import SidebarMenuSubActiveButton from "./SidebarMenuSubActiveButton";
import { useContext } from 'react';
import { UserContext } from "@/lib/providers";

// const AiMagicIcon = (props: React.SVGProps<SVGSVGElement>) => (
//   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width={44} height={44} color={"#CBC8C8"} fill={"none"} {...props}>
//     <path d="M10 7L9.48415 8.39405C8.80774 10.222 8.46953 11.136 7.80278 11.8028C7.13603 12.4695 6.22204 12.8077 4.39405 13.4842L3 14L4.39405 14.5158C6.22204 15.1923 7.13603 15.5305 7.80278 16.1972C8.46953 16.864 8.80774 17.778 9.48415 19.6059L10 21L10.5158 19.6059C11.1923 17.778 11.5305 16.864 12.1972 16.1972C12.864 15.5305 13.778 15.1923 15.6059 14.5158L17 14L15.6059 13.4842C13.778 12.8077 12.864 12.4695 12.1972 11.8028C11.5305 11.136 11.1923 10.222 10.5158 8.39405L10 7Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
//     <path d="M18 3L17.7789 3.59745C17.489 4.38087 17.3441 4.77259 17.0583 5.05833C16.7726 5.34408 16.3809 5.48903 15.5975 5.77892L15 6L15.5975 6.22108C16.3809 6.51097 16.7726 6.65592 17.0583 6.94167C17.3441 7.22741 17.489 7.61913 17.7789 8.40255L18 9L18.2211 8.40255C18.511 7.61913 18.6559 7.22741 18.9417 6.94166C19.2274 6.65592 19.6191 6.51097 20.4025 6.22108L21 6L20.4025 5.77892C19.6191 5.48903 19.2274 5.34408 18.9417 5.05833C18.6559 4.77259 18.511 4.38087 18.2211 3.59745L18 3Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
//   </svg>
// );
// Menu items.
// const items = [
//   {
//     title: "Dashboard",
//     url: "/dashboard",
//     icon: Home,
//   },
//   {
//     title: "Talent finder",
//     url: "/talent-finder",
//     icon: AiMagicIcon,
//     items: [
//       {
//         title: "search 1",
//         url: "#",
//       },
//       {
//         title: "search 2",
//         url: "#",
//       },
//     ],
//   },
//   {
//     title: "Source",
//     url: "/source",
//     icon: Unplug,
//   },
// ]

const user = {
  name: "John",
  email: "JOHN@example.com",
  // avatar: CircleUser,
}

export function AppSidebar() {
  const { items } = useContext(UserContext);
  // console.log(items)

  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          {/* <SidebarGroupLabel>Application</SidebarGroupLabel> */}
          <SidebarMenuButton size="lg" asChild>
            <a href="#" className="mb-5 mt-2">
              <Image src="/DataCurvelogo.png" alt="DataCurve" width={150} height={100} />
            </a>
          </SidebarMenuButton>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  {/* <SidebarMenuButton asChild isActive={item.url === pathname}>
                    <Link href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton> */}
                  <SidebarActiveButton url={item.url} title={item.title}>{item.icon ? React.createElement(item.icon) : null}</SidebarActiveButton>
                  {item.items?.length ? (
                    <>
                      {item.items.map((item) => (
                        <SidebarMenuSubItem key={item.title}>
                          {/* <SidebarMenuSubButton asChild isActive={item.isActive}>
                            <a href={item.url}>{item.title}</a>
                          </SidebarMenuSubButton> */}
                          <SidebarMenuSubActiveButton url={item.url} title={item.title} />
                        </SidebarMenuSubItem>
                      ))}
                    </>
                  ) : null}
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={user}>
          {React.createElement(CircleUser)}
        </NavUser>
      </SidebarFooter>
    </Sidebar>
  )
}
