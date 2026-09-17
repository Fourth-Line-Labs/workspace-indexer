namespace Fixture.Web;

// Every line here is an object initializer that a secret scanner has read as a
// credential. The eight files purged from a real index were exactly this shape:
// the key matches on `token` or `secret`, and the value is an *expression* the
// scanner could not see through because C#'s `?.` puts a `?` where a dot was
// expected.
public class Options
{
    private readonly int _maxTokens = 4096;

    public Options Build(Options options, Domain domain) => new()
    {
        MaxOutputTokens = options?.MaxOutputTokens ?? _maxTokens,
        ClaudeSecretRef = domain?.ClaudeSecretRef,
        ApiKey = settings.VoyageApiKey,
        ConnectionString = builder.Configuration["AppConfig"],
        AuthMode = AuthMode.TrustedLocal,
        Credentials = "docker-hub-credentials",
        PasswordTemplate = "${DB_PASSWORD}",
    };

    public int MaxOutputTokens { get; init; }

    public string ClaudeSecretRef { get; init; }

    public string ApiKey { get; init; }

    public string ConnectionString { get; init; }

    public AuthMode AuthMode { get; init; }

    public string Credentials { get; init; }

    public string PasswordTemplate { get; init; }
}
