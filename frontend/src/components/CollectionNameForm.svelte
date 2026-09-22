<script lang="ts">
  import { untrack } from "svelte";
  let { submit, initial = "", renaming = false, disabled = false }: {
    submit: (name: string) => Promise<void>;
    initial?: string;
    renaming?: boolean;
    disabled?: boolean;
  } = $props();
  let name = $state(untrack(() => initial));
  let pending = $state(false);
  let error = $state("");

  async function save(event: SubmitEvent) {
    event.preventDefault();
    if (pending || disabled) return;
    const trimmed = name.trim();
    if (!trimmed || Array.from(trimmed).length > 100) {
      error = "Use a collection name of 1–100 characters.";
      return;
    }
    pending = true;
    error = "";
    try {
      await submit(trimmed);
      if (!renaming) name = "";
    } catch (err) {
      error = err instanceof Error ? err.message : "Could not save collection";
    } finally {
      pending = false;
    }
  }
</script>

<form onsubmit={save} class="flex flex-wrap items-center gap-2">
  <input aria-label={renaming ? "New collection name" : "Collection name"} placeholder="Collection name" bind:value={name} disabled={pending || disabled} class="min-w-0 flex-1 px-3 py-2 bg-surface border border-line-2 rounded-lg" />
  <button type="submit" disabled={pending || disabled} class="px-3 py-2 bg-ink text-white rounded-lg cursor-pointer disabled:opacity-50">{pending ? "Saving…" : renaming ? "Save name" : "Create collection"}</button>
  {#if error}<p role="alert" class="w-full text-bad text-sm">{error}</p>{/if}
</form>
