# Security policy

Please report suspected vulnerabilities through GitHub private vulnerability
reporting under the repository Security tab rather than opening a public issue.

The policy server is intended for local or trusted-network use. It has no
built-in authentication or TLS and binds to `127.0.0.1` by default. If remote
access is required, place it behind an authenticated proxy or firewall and
restrict the exposed port.

Never commit access tokens, proxy credentials, private dataset paths, model
weights, or experiment outputs. Rotate any credential immediately if it appears
in Git history; deleting it only from the current working tree is insufficient.
