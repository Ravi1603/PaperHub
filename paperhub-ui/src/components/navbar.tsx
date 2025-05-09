"use client";
import React from "react";
import {
  Navbar as MTNavbar,
  Collapse,
  Button,
  IconButton,
  Typography,
} from "@material-tailwind/react";
import {
  RectangleStackIcon,
  UserCircleIcon,
  CommandLineIcon,
  XMarkIcon,
  Bars3Icon,
} from "@heroicons/react/24/solid";

const NAV_MENU = [
  { name: "Page", icon: RectangleStackIcon },
  { name: "Account", icon: UserCircleIcon },
  { name: "Docs", icon: CommandLineIcon, href: "https://www.material-tailwind.com/docs/react/installation" },
];

interface NavItemProps {
  children: React.ReactNode;
  href?: string;
}

function NavItem({ children, href }: NavItemProps) {
  return (
    <li>
      <Typography
        as="a"
        href={href || "#"}
        target={href ? "_blank" : "_self"}
        variant="paragraph"
        className="flex items-center gap-2 font-medium text-[#1e1e1e] hover:text-gray-800 transition-colors"
      >
        {children}
      </Typography>
    </li>
  );
}

export function Navbar() {
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen((cur) => !cur);

  React.useEffect(() => {
    window.addEventListener("resize", () => window.innerWidth >= 960 && setOpen(false));
  }, []);

  return (
    <MTNavbar
      shadow={false}
      fullWidth
      className="sticky top-0 z-50 border-0 bg-gradient-to-br from-[#bbd0ff] via-[#b8c0ff] to-[#ffd6ff] text-black"
    >
      <div className="container mx-auto flex items-center justify-between py-2">
        <Typography
          as="a"
          href="/"
          className="text-xl font-extrabold text-[#2b2b2b] tracking-wide"
        >
          PaperHub
        </Typography>

        <ul className="ml-10 hidden items-center gap-8 lg:flex">
          {NAV_MENU.map(({ name, icon: Icon, href }) => (
            <NavItem key={name} href={href}>
              <Icon className="h-5 w-5" />
              {name}
            </NavItem>
          ))}
        </ul>

        <div className="hidden items-center gap-2 lg:flex">
          <Button variant="text" className="text-[#2b2b2b] hover:text-black">Sign In</Button>
          <Button color="gray" className="bg-white text-black hover:bg-gray-100 shadow-md">
            Blocks
          </Button>
        </div>

        <IconButton
          variant="text"
          onClick={handleOpen}
          className="ml-auto inline-block lg:hidden text-black"
        >
          {open ? (
            <XMarkIcon strokeWidth={2} className="h-6 w-6" />
          ) : (
            <Bars3Icon strokeWidth={2} className="h-6 w-6" />
          )}
        </IconButton>
      </div>

      <Collapse open={open}>
        <div className="container mx-auto mt-3 border-t border-gray-200 px-2 pt-4">
          <ul className="flex flex-col gap-4">
            {NAV_MENU.map(({ name, icon: Icon, href }) => (
              <NavItem key={name} href={href}>
                <Icon className="h-5 w-5" />
                {name}
              </NavItem>
            ))}
          </ul>
          <div className="mt-6 mb-4 flex items-center gap-2">
            <Button variant="text" className="text-[#2b2b2b] hover:text-black">Sign In</Button>
            <Button color="gray" className="bg-white text-black hover:bg-gray-100 shadow-md">
              Blocks
            </Button>
          </div>
        </div>
      </Collapse>
    </MTNavbar>
  );
}

export default Navbar;
