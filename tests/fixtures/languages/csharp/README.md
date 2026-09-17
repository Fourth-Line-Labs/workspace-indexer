# C# fixtures

`using` names a namespace, not a path, so resolution joins against the
namespaces declared in the same repository. These files exercise the shapes
that decide what it can follow.

| file | what it is here for |
|---|---|
| `Data/Repo.cs`, `Data/Context.cs`, `Data/Helpers.cs` | **three** files declaring **one** namespace — a using naming it reaches all three, which is why resolution returns a list rather than a path. `Helpers.cs` is also the type a `using static` names |
| `Web/Startup.cs` | five directive forms at once: framework (`System`, `System.Text.Json`), a package (`Azure.Identity`), first-party (`Fixture.Data`), `using static` (declined — it names a type), and `using Alias = ...` (the target is the qualified name, not the alias) |
| `Web/Shared.cs` | `global using`, recorded under its own kind; it resolves like any other using for the declaring file, and the compilation-unit propagation is deliberately not modelled |
| `Legacy/Block.cs` | the block form, nested — `Fixture.Legacy` and `Fixture.Legacy.Inner` are both declared, because `using Inner` reaches nothing |
| `Legacy/Two.cs` | two namespaces in one file: absent from both measured corpora, which is exactly why it is here — the schema allows it and nothing else exercises it |
| `Web/Options.cs` | object initializers a secret scanner has read as credentials — the `?.` and `??` shape that purged eight files from a real index, plus a dotted attribute, a bracket expression, an enum member, a credential's *name* and a template. It exists to be **indexed** |

Ten declarations across eight files is the number to check when this changes:
three for `Fixture.Data`, **two** for `Fixture.Web` (`Startup.cs` and
`Options.cs`), one for `Fixture.Web.Shared`, two from the nested block, two
from the two-namespace file.
