import { randomColor, dateString } from "../utils";
import Logo from "./Logo";

const logoColors = [randomColor(), randomColor()];

interface HeaderProps {
  date: string | null;
  onDateChange: (date: string | null) => void;
}

const Header = ({ date, onDateChange }: HeaderProps) => {
  return (
    <header>
      <div className="logo">
        <Logo colors={logoColors} />
      </div>
      <h1>
        <a href="https://github.com/gradbench/gradbench">GradBench</a>{" "}
      </h1>
      {date !== null &&
        <nav>
          <button
            disabled={date === undefined}
            onClick={() => {
              if (date === undefined) return;
              const d = new Date(date);
              d.setDate(d.getDate() - 1);
              onDateChange(dateString(d));
            }}
          >
            ◀
          </button>{" "}
          <input
            type="date"
            value={date ?? ""}
            onChange={(e) => onDateChange(e.target.value)}
          />{" "}
          <button
            disabled={date === undefined}
            onClick={() => {
              if (date === undefined) return;
              const d = new Date(date);
              d.setDate(d.getDate() + 1);
              onDateChange(dateString(d));
            }}
          >
            ▶
          </button>
        </nav>
      }
    </header>
  );
}

export default Header;
