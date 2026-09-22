import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm, copyFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const repo = fileURLToPath(new URL("../", import.meta.url));
const html = '<div id="app"></div><script type="module" crossorigin src="/assets/app-123.js"></script><link rel="stylesheet" href="/assets/app-123.css">';

async function fixture(t, files = {}) {
  const dir = await mkdtemp(join(tmpdir(), "atelier-build-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const [name, content] of Object.entries(files)) {
    const path = join(dir, name);
    await mkdir(join(path, ".."), { recursive: true });
    await writeFile(path, content);
  }
  return dir;
}

for (const [name, files] of Object.entries({
  missing: {},
  placeholder: { "index.html": "<h1>Atelier frontend not built yet</h1>" },
  source: { "index.html": '<div id="app"></div><script type="module" src="/src/main.ts"></script>' },
  "missing assets": { "index.html": html },
  "missing css": { "index.html": html, "assets/app-123.js": "console.log('app')" },
  "empty js": { "index.html": html, "assets/app-123.js": "", "assets/app-123.css": "body{}" },
})) {
  test(`validator rejects ${name}`, async (t) => {
    const { validateAtelierBuild } = await import("./validate_atelier_build.mjs");
    await assert.rejects(validateAtelierBuild(await fixture(t, files)), /Atelier build/);
  });
}

test("validator accepts emitted index, JS and CSS", async (t) => {
  const { validateAtelierBuild } = await import("./validate_atelier_build.mjs");
  await validateAtelierBuild(await fixture(t, {
    "index.html": html, "assets/app-123.js": "console.log('app')", "assets/app-123.css": "body{}",
  }));
});

test("smoke fails rather than skipping missing dist", async (t) => {
  const dir = await fixture(t);
  // CI runs script regressions before installing Playwright: invalid assets
  // must fail at the build gate, not while loading browser dependencies.
  for (const name of ["ui_smoke_test_atelier.mjs", "validate_atelier_build.mjs"]) {
    await copyFile(join(repo, "scripts", name), join(dir, name));
  }
  const result = spawnSync(process.execPath, [join(dir, "ui_smoke_test_atelier.mjs")], {
    env: { ...process.env, ATELIER_DIST: dir }, encoding: "utf8",
  });
  assert.notEqual(result.status, 0, result.stdout);
  assert.match(result.stderr, /Atelier build/);
});

test("release skip-build still rejects absent dist before go build", async (t) => {
  const dir = await fixture(t, {
    "libs/.keep": "", "bin/go": "#!/bin/sh\necho GO_BUILD_MUST_NOT_RUN >&2\nexit 1\n",
  });
  const { chmod } = await import("node:fs/promises");
  await chmod(join(dir, "bin/go"), 0o755);
  await mkdir(join(dir, "scripts"));
  await copyFile(join(repo, "scripts/package_release.sh"), join(dir, "scripts/package_release.sh"));
  await copyFile(join(repo, "scripts/validate_atelier_build.mjs"), join(dir, "scripts/validate_atelier_build.mjs"));
  const result = spawnSync("bash", [join(dir, "scripts/package_release.sh"), "test"], {
    env: { ...process.env, PATH: `${join(dir, "bin")}:${process.env.PATH}`, IMGSEARCH_SKIP_FRONTEND_BUILD: "1", IMGSEARCH_LLAMA_LIB_DIR: join(dir, "libs") }, encoding: "utf8",
  });
  assert.notEqual(result.status, 0);
  assert.doesNotMatch(result.stderr, /GO_BUILD_MUST_NOT_RUN/);
  assert.match(result.stderr, /Atelier build/);
});

test("delivery and UI wiring require Atelier only", async () => {
  const { readFile } = await import("node:fs/promises");
  const read = (path) => readFile(join(repo, path), "utf8");
  const container = await read("Containerfile.cuda");
  assert.match(container, /FROM .*node:24.* AS frontend/);
  assert.match(container, /npm ci[^\n]*\nRUN npm run build/);
  assert.match(container, /COPY --from=frontend \/src\/internal\/webui\/atelier\/dist\//);
  assert.ok(container.indexOf("COPY --from=frontend") < container.indexOf("RUN go build"));
  assert.match(await read("frontend/package.json"), /validate_atelier_build.mjs/);
  const dockerignore = await read(".dockerignore");
  assert.match(dockerignore, /^\*\*\/node_modules\/$/m);
  assert.match(dockerignore, /^\/internal\/webui\/atelier\/dist\/$/m);
  for (const path of ["frontend/src/components/Rail.svelte", "frontend/vite.config.ts", "package.json"]) {
    assert.doesNotMatch(await read(path), /\/legacy|test:legacy|ui_smoke_test\.mjs/);
  }
  const ci = await read(".github/workflows/ci.yml");
  assert.ok(ci.indexOf("npm run build") < ci.indexOf("go test ./..."), "Go tests must exercise fresh embeds in CI");
});
