# NAS container inventory guide

Part of #38 (docker-compose consolidation). Run this **on the NAS itself**
(SSH in, or use the Container Station / QTS terminal) — an agent working from
this repo does not have network access to the NAS, so this inventory has to
be gathered by hand and pasted back in (e.g. into a PR description, a
comment on #38, or a scratch file) before the `docker-compose.yml` can be
written.

## 1. List all running (and stopped) containers

```bash
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
```

Note which containers are actually in scope for #38 (at minimum: dnsmasq,
nginx, Portainer — plus anything else you want reproducible).

## 2. Dump full config per container

For each container name from step 1, run:

```bash
docker inspect <container_name> > <container_name>.json
```

Each dump has everything needed to reconstruct a compose service, in
particular:

| What you need | Where in `docker inspect` output |
|---|---|
| Image (with tag/digest) | `.Config.Image` |
| Command / entrypoint override | `.Config.Cmd`, `.Config.Entrypoint` |
| Environment variables | `.Config.Env` |
| Volume/bind mounts | `.Mounts` (source, destination, mode) |
| Port mappings | `.NetworkSettings.Ports` (or `.HostConfig.PortBindings`) |
| Network mode | `.HostConfig.NetworkMode` (e.g. `host` for dnsmasq) |
| Restart policy | `.HostConfig.RestartPolicy` |
| Extra `--cap-add` / `--privileged` / `--device` flags | `.HostConfig.CapAdd`, `.HostConfig.Privileged`, `.HostConfig.Devices` |

If `jq` is available, this pulls just the relevant fields per container:

```bash
docker inspect <container_name> | jq '.[0] | {
  Image: .Config.Image,
  Env: .Config.Env,
  Mounts: .Mounts,
  Ports: .NetworkSettings.Ports,
  NetworkMode: .HostConfig.NetworkMode,
  RestartPolicy: .HostConfig.RestartPolicy,
  CapAdd: .HostConfig.CapAdd,
  Privileged: .HostConfig.Privileged
}'
```

## 3. Grab any config files mounted from NAS-local paths

For each bind mount found in `.Mounts` above (e.g. `dnsmasq.conf`, nginx
site configs), copy the file content too — these need to move into the repo
alongside the compose file per #38's task list.

```bash
cat /path/on/nas/dnsmasq.conf
```

## 4. Bring it back

Paste the `docker inspect` output (or the `jq`-filtered version) and any
config file contents into the #38 issue thread, or drop them as files under
`docs/nas/inventory/` in a branch — either way, that's what turns into the
actual `docker-compose.yml` service definitions and externalized config
files.

## Later: giving an agent direct access instead

If at some point you want an agent to run this inventory itself rather than
copy-pasting output by hand, that means giving it SSH (or Docker API/socket)
access to the NAS — worth its own follow-up issue if/when you want it, since
it's a real trust boundary (the agent would be able to run arbitrary
commands on the NAS) rather than something to bolt on here.
