<script lang="ts">
  import { onMount, tick, untrack } from "svelte";
  import { createCollection, listCollections, setCollectionMembership } from "../lib/api";
  import { actionTarget } from "../lib/mediaActions";
  import { collectionPickerTarget, collectionPickerOpener, refreshCollections } from "../lib/stores";
  import { focusTrap } from "../lib/focus";
  import type { Collection, Pin } from "../lib/types";
  import CollectionNameForm from "./CollectionNameForm.svelte";

  let { pin }: { pin: Pin } = $props();
  const opener = collectionPickerOpener;
  const target = untrack(() => actionTarget(pin));
  let collections = $state<Collection[]>([]);
  let memberships = $state<number[]>([]);
  let loading = $state(true);
  let loaded = $state(false);
  let pending = $state(false);
  let error = $state("");
  let notice = $state("");
  let active = true;
  let controller: AbortController | undefined;

  function close() { collectionPickerTarget.set(null); }
  function onKey(event: KeyboardEvent) {
    // Do not let any picker keystroke reach Feed/lightbox window handlers.
    event.stopPropagation();
    if (event.key === "Escape") { event.preventDefault(); close(); }
  }
  async function load() {
    controller?.abort();
    const ac = new AbortController();
    controller = ac;
    loading = true;
    loaded = false;
    error = "";
    try {
      const [all, saved] = await Promise.all([
        listCollections({ signal: ac.signal }),
        listCollections({ target, signal: ac.signal }),
      ]);
      if (!active || ac.signal.aborted) return;
      loaded = true;
      collections = all.collections;
      memberships = saved.collections.map((c) => c.id);
    } catch (err) {
      if (active && !ac.signal.aborted) error = err instanceof Error ? err.message : "Could not load collections";
    } finally {
      if (active && !ac.signal.aborted) loading = false;
    }
  }
  onMount(() => {
    void load();
    return () => {
      active = false;
      controller?.abort();
      // Wait until the underlying view is no longer inert. A removed card may
      // no longer exist; return to collection navigation in that case.
      void tick().then(() => {
        if (opener?.isConnected) opener.focus();
        else document.querySelector<HTMLElement>('[data-rail-item="collections"]')?.focus();
      });
    };
  });

  async function save(id: number, selected: boolean) {
    await setCollectionMembership(id, target, selected);
    refreshCollections();
    if (!active) return;
    memberships = selected ? [...memberships, id] : memberships.filter((m) => m !== id);
    notice = selected ? "Saved to collection." : "Removed from collection. Original media kept.";
  }
  async function toggle(id: number) {
    if (pending) return;
    pending = true;
    error = "";
    notice = "";
    try {
      await save(id, !memberships.includes(id));
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not update membership";
    } finally {
      if (active) pending = false;
    }
  }
  async function create(name: string) {
    pending = true;
    error = "";
    notice = "";
    try {
      const collection = await createCollection(name);
      refreshCollections();
      if (active) collections = [collection, ...collections];
      // If saving fails the new collection remains available for retry.
      await save(collection.id, true);
    } finally {
      if (active) pending = false;
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="fixed inset-0 z-[1600] bg-black/55 grid place-items-center p-4" onkeydown={onKey}>
  <div data-collection-picker role="dialog" aria-modal="true" aria-label="Save to collection" tabindex="-1" use:focusTrap class="bg-surface text-ink w-full max-w-md max-h-[90dvh] overflow-auto p-5 rounded-card shadow-elev">
    <div class="flex items-center justify-between gap-3 mb-3">
      <h2 class="text-xl font-display">Save to collection</h2>
      <button aria-label="Close collection picker" class="px-3 py-2 cursor-pointer" onclick={close}>×</button>
    </div>
    <p class="text-sm text-muted break-all mb-4">{pin.title}</p>
    <p class="text-sm text-muted mb-4">Check to save; uncheck to remove from a collection. Original media stays in your library.</p>
    {#if loading}
      <p role="status">Loading collections…</p>
    {:else if loaded}
      {#if collections.length === 0 && !error}<p class="mb-4">No collections yet.</p>{/if}
      <div class="grid gap-2 mb-4">
        {#each collections as collection (collection.id)}
          <label class="flex items-center gap-3 p-2 border border-line rounded-lg break-all">
            <input type="checkbox" checked={memberships.includes(collection.id)} aria-disabled={pending} onclick={(event) => { event.preventDefault(); void toggle(collection.id); }} />
            {collection.name}
          </label>
        {/each}
      </div>
      <CollectionNameForm submit={create} disabled={pending} />
    {/if}
    {#if error}<p role="alert" class="text-bad mt-3">{error}</p>{#if !loaded}<button onclick={load} class="mt-2 cursor-pointer">Retry</button>{/if}{/if}
    <p role="status" class="text-sm mt-3">{pending ? "Saving…" : notice}</p>
  </div>
</div>
