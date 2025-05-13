
/** Return a YYYY-MM-DD date string from a `Date` object. */
export const dateString = (date: Date): string => date.toISOString().split("T")[0];

/**
 * Return `date` if it is a valid YYYY-MM-DD date string, otherwise `undefined`.
 */
export const parseDate = (date: string | null | undefined): string | undefined => {
  if (date === null || date === undefined) return undefined;
  try {
    if (dateString(new Date(date)) === date) {
      return date;
    }
  } catch (_) {
    return undefined;
  }
};

export const randomColor = () => {
  return `#${Math.floor(Math.random() * 16777215).toString(16)}`;
};
