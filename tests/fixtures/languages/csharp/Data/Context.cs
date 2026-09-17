namespace Fixture.Data;

// A second file declaring the *same* namespace -- `Helpers.cs` is a third --
// so a using naming it reaches all three, which is why resolution returns a
// list and not a path.
public class Context
{
    public string Name => "fixture";
}
