global using Fixture.Data;

namespace Fixture.Web.Shared;

// `global using` applies to the whole compilation unit. This rung records it
// under its own kind and resolves it like any other using for the declaring
// file only -- the propagation is deliberately not modelled.
public class SharedThing
{
}
