namespace Fixture.Legacy
{
    // The block form, and nested. `using Inner` reaches nothing: the name a
    // using has to spell is `Fixture.Legacy.Inner`.
    namespace Inner
    {
        public class Nested
        {
        }
    }
}
