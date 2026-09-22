<script lang="ts">
  import { onMount, tick, untrack } from "svelte";
  import { get } from "svelte/store";
  import { createCollection, deleteCollection, listCollectionSummaries, renameCollection } from "../lib/api";
  import { collectionEpoch, includeNSFW, refreshCollections, setCollection } from "../lib/stores";
  import { collectionPresentation } from "../lib/collectionPresentation";
  import type { Collection, CollectionSummary } from "../lib/types";
  import CollectionCovers from "./CollectionCovers.svelte";
  import CollectionNameForm from "./CollectionNameForm.svelte";
  import ConfirmDialog from "./ConfirmDialog.svelte";

  let collections = $state<CollectionSummary[]>([]);
  let loading = $state(true);
  let pending = $state(false);
  let error = $state("");
  let renaming = $state<number | null>(null);
  let deleting = $state<Collection | null>(null);
  let managing = $state<number | null>(null);
  let active = true;
  let request = 0;
  let controller: AbortController | undefined;
  let loadedScope = $state("");
  const scope = $derived(`${$collectionEpoch}:${$includeNSFW}`);
  // Gate in the render as well as clearing on fetch: no stale unsafe covers,
  // even before effects run or if the replacement request fails.
  const visibleCollections = $derived(loadedScope === scope ? collections : []);
  let focusAfterLoad: number | "create" | null = null;

  function focusManage(id: number) {
    document.querySelector<HTMLButtonElement>(`[data-collection-manage="${id}"]`)?.focus();
  }
  function closeMenu() {
    if (managing !== null) focusManage(managing);
    managing = null;
  }
  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape" && managing !== null) {
      event.preventDefault();
      closeMenu();
    }
  }
  function onClick(event: MouseEvent) {
    if (!(event.target instanceof Element) || !event.target.closest("[data-collection-management]")) managing = null;
  }
  async function load() {
    const token = ++request;
    controller?.abort();
    const ac = new AbortController();
    controller = ac;
    const include = get(includeNSFW);
    const epoch = get(collectionEpoch);
    const key = `${epoch}:${include}`;
    const current = () => active && token === request && !ac.signal.aborted && include === get(includeNSFW) && epoch === get(collectionEpoch);
    loading = true;
    error = "";
    collections = [];
    managing = null;
    try {
      const response = await listCollectionSummaries({ includeNSFW: include, signal: ac.signal });
      if (current()) {
        collections = response.collections;
        loadedScope = key;
      }
    } catch (err) {
      if (current()) error = err instanceof Error ? err.message : "Could not load collections";
    } finally {
      if (current()) {
        loading = false;
        if (focusAfterLoad !== null) {
          const id = focusAfterLoad;
          focusAfterLoad = null;
          void tick().then(() => {
            if (!current()) return;
            if (id === "create") document.querySelector<HTMLInputElement>('[aria-label="Collection name"]')?.focus();
            else focusManage(id);
          });
        }
      }
    }
  }
  $effect(() => {
    void scope;
    untrack(() => void load());
  });
  onMount(() => () => { active = false; controller?.abort(); });

  async function create(name: string) {
    await createCollection(name);
    refreshCollections();
  }
  async function rename(id: number, name: string) {
    await renameCollection(id, name);
    if (active) { renaming = null; focusAfterLoad = id; }
    refreshCollections();
  }
  function startRename(id: number) {
    closeMenu();
    renaming = id;
    void tick().then(() => document.querySelector<HTMLInputElement>('[aria-label="New collection name"]')?.focus());
  }
  function cancelRename(id: number) {
    renaming = null;
    void tick().then(() => focusManage(id));
  }
  async function remove() {
    const item = deleting;
    deleting = null;
    if (!item || pending) return;
    pending = true;
    error = "";
    try {
      await deleteCollection(item.id);
      if (active) focusAfterLoad = "create";
      refreshCollections();
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not delete collection";
    } finally {
      if (active) pending = false;
    }
  }
  const button = "px-3 py-2 border border-line rounded-lg cursor-pointer disabled:opacity-50";
</script>

<svelte:window onkeydown={onKey} onclick={onClick} />

<section data-collections-pane class="px-5 sm:px-9 py-5 max-w-5xl w-full">
  <p class="text-muted mb-4">Saved media in this library. Deleting a collection never deletes the original media.</p>
  <CollectionNameForm submit={create} disabled={loading || pending} />
  {#if loading}<p role="status" class="mt-4">Loading collections…</p>{/if}
  {#if error}<p role="alert" class="text-bad mt-4">{error}</p><button class={button} onclick={load} disabled={pending}>Retry</button>{/if}
  {#if !loading && !error && visibleCollections.length === 0}<p class="mt-4 text-muted">No collections yet. Create one to start saving media.</p>{/if}
  <ul class="list-none p-0 mt-5 grid md:grid-cols-2 gap-3">
    {#each visibleCollections as collection (collection.id)}
      {@const presentation = collectionPresentation(collection.item_count, collection.visible_count)}
      <li data-collection-card={collection.id} class="relative min-w-0 bg-surface border border-line rounded-card flex items-stretch">
        {#if renaming === collection.id}
          <div class="p-4 min-w-0 w-full">
            <CollectionNameForm initial={collection.name} renaming submit={(name) => rename(collection.id, name)} />
            <button class={`${button} mt-2`} onclick={() => cancelRename(collection.id)}>Cancel rename</button>
          </div>
        {:else}
          <button class="flex-1 min-w-0 flex items-center gap-3 p-4 text-left rounded-card cursor-pointer hover:bg-surface-2 focus-visible:outline-accent" aria-label={`Browse ${collection.name}`} aria-describedby={`collection-count-${collection.id}${presentation.message ? ` collection-status-${collection.id}` : ""}`} onclick={() => setCollection(collection.id)}>
            <CollectionCovers covers={collection.covers} />
            <span class="min-w-0">
              <span class="block font-medium break-words [overflow-wrap:anywhere]">{collection.name}</span>
              <span id={`collection-count-${collection.id}`} class="block text-sm text-muted mt-1">{presentation.count}</span>
              {#if presentation.message}<span id={`collection-status-${collection.id}`} class="block text-xs text-muted-2 mt-1">{presentation.message}</span>{/if}
            </span>
          </button>
          <div data-collection-management class="shrink-0 p-2 pl-0">
            <button data-collection-manage={collection.id} class="w-9 h-9 rounded-lg cursor-pointer hover:bg-surface-2 disabled:opacity-50" disabled={pending} aria-label={`Manage ${collection.name}`} aria-expanded={managing === collection.id} aria-controls={`collection-manage-${collection.id}`} onclick={() => managing = managing === collection.id ? null : collection.id}>⋯</button>
            {#if managing === collection.id}
              <div id={`collection-manage-${collection.id}`} class="absolute right-2 top-12 z-10 grid gap-1 p-2 bg-surface border border-line rounded-lg shadow-elev">
                <button class={button} aria-label={`Rename ${collection.name}`} onclick={() => startRename(collection.id)}>Rename</button>
                <button class={`${button} text-bad`} aria-label={`Delete ${collection.name}`} onclick={() => { closeMenu(); deleting = collection; }}>Delete collection</button>
              </div>
            {/if}
          </div>
        {/if}
      </li>
    {/each}
  </ul>
</section>
{#if deleting}
  <ConfirmDialog title="Delete collection?" detail={deleting.name} note="Only the collection and its memberships will be removed. Original images and videos stay in your library." confirmLabel="Delete collection" onconfirm={() => void remove()} oncancel={() => deleting = null} />
{/if}
