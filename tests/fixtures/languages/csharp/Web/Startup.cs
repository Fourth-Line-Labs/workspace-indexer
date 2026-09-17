using System;
using System.Text.Json;
using Azure.Identity;
using Fixture.Data;
using static Fixture.Data.Helpers;
using Alias = Fixture.Legacy.Inner;

namespace Fixture.Web;

// Five directive forms in one file:
//   System / System.Text.Json  framework, correctly unresolved
//   Azure.Identity             a package, unclassified until a manifest reader
//   Fixture.Data               first-party, resolves to all three files that
//                              declare it: Repo.cs, Context.cs, Helpers.cs
//   using static ...           names a type; declined, not resolved
//   using Alias = ...          the target is the qualified name, not the alias
public class Startup
{
    public void Configure() => Console.WriteLine(Double(2));
}
