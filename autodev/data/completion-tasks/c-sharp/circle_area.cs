abstract class Shape
{
    abstract public double CalcArea();
}

class Circle : Shape
{
    private double radius;
    public Circle(double radius)
    {
        this.radius = radius;
    }
    public override double CalcArea()
    {
        <todo>
    }
}
