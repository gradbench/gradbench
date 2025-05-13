import { dateString, randomColor } from "../utils";
import Logo from "./Logo";

const logoColors = [randomColor(), randomColor()];

interface DatePickerProps {
  date: string | null;
  onDateChange: (date: string) => void;
}

const DatePicker = ({ date, onDateChange }: DatePickerProps) => {
  const next = () => {
    if (date === null) return;
    const d = new Date(date);
    d.setDate(d.getDate() + 1);
    onDateChange(dateString(d));
  };
  const previus = () => {
    if (date === null) return;
    const d = new Date(date);
    d.setDate(d.getDate() - 1);
    onDateChange(dateString(d));
  };

  return (
    <nav className="date-picker">
      <button
        className="date-picker__button"
        disabled={date === null}
        onClick={previus}
      >
        {" "}
        ◀{" "}
      </button>
      <input
        className="date-picker__input"
        type="date"
        value={date ?? ""}
        onChange={(e) => onDateChange(e.target.value)}
      />
      <button
        className="date-picker__button"
        disabled={date === null}
        onClick={next}
      >
        {" "}
        ▶{" "}
      </button>
    </nav>
  );
};

interface HeaderProps {
  date: string | null;
  onDateChange: (date: string) => void;
}

const Header = ({ date, onDateChange }: HeaderProps) => {
  return (
    <header className="header">
      <div className="header__logo">
        <Logo colors={logoColors} />
      </div>
      <h1 className="header__title">
        <a href="https://github.com/gradbench/gradbench">GradBench</a>
      </h1>
      {date !== null && <DatePicker date={date} onDateChange={onDateChange} />}
    </header>
  );
};

export default Header;
