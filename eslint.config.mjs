import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "warn", // Set 'no-explicit-any' rule to 'warn'
      "@typescript-eslint/no-unused-vars": "warn" // Set 'no-explicit-any' rule to 'warn'
    }
  }
];

export default eslintConfig;
