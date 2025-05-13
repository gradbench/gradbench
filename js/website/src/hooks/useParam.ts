import { useEffect, useState } from "react";

type useParamHook = [
  value: string | null,
  setValue: (value: string | null) => void,
];

export function useParam(
  key: string,
  defaultValue: string | null,
): useParamHook {
  const params = new URL(window.location.href).searchParams;
  const [value, setValue] = useState(params.get(key) ?? defaultValue);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (value === null) {
      url.searchParams.delete(key);
    } else {
      url.searchParams.set(key, value);
    }
    window.history.pushState(null, "", url.href);
  }, [value, key]);

  return [value, setValue];
}
