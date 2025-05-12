const vec2 = (x: number, y: number): [number, number] => [x, y];

const str = (points: [number, number][]) =>
  points.map(([x, y]) => `${x},${y}`).join(" ");

const size = 500;
const gap = 50;
const thickness = 50;

const height = size / 2 - (gap + gap / 2);
const width = height / (Math.sqrt(3) / 2);

const grad = () => {
  return [
    vec2(size / 2 + width / 2, gap),
    vec2(size / 2 - width / 2, gap),
    vec2(size / 2, size / 2 - gap / 2),
  ];
};

const bench = () => {
  return [
    vec2(size - gap, size - gap),
    vec2(size - gap, size / 2 + gap / 2),
    vec2(gap, size / 2 + gap / 2),
    vec2(gap, size - gap),
    vec2(gap + thickness, size - gap),
    vec2(gap + thickness, size / 2 + gap / 2 + thickness),
    vec2(size - (gap + thickness), size / 2 + gap / 2 + thickness),
    vec2(size - (gap + thickness), size - gap),
  ];
};

const Logo = (props: { colors: string[]; gradientId?: string }) => {
  const colors =
    props.colors.length > 1 ? props.colors : [props.colors[0], props.colors[0]];
  const gradientId = props.gradientId ?? "bggradient";
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      version="1.1"
      viewBox={`0 0 ${size} ${size}`}
      width="100%"
      height="100%"
    >
      <defs>
        <linearGradient id={gradientId}>
          {colors.map((color, index) => {
            const proportion = index / (colors.length - 1);
            const offset = `${proportion * 100}%`;
            return <stop key={offset} offset={offset} stopColor={color} />;
          })}
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${gradientId})`} rx={gap} />
      <polygon fill="#fff" points={str(grad())} />
      <polygon fill="#fff" points={str(bench())} />
    </svg>
  );
};

export default Logo;
