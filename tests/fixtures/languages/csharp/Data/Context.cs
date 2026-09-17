namespace Fixture.Data;

// A second file declaring the *same* namespace: a using naming it reaches both,
// which is why resolution returns a list and not a path.
public class Context
{
    public string Name => "fixture";
}
