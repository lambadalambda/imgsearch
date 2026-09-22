<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { createCollection, deleteCollection, listCollections, renameCollection } from "../lib/api";
  import { collectionEpoch, refreshCollections, setCollection } from "../lib/stores";
  import type { Collection } from "../lib/types";
  import CollectionNameForm from "./CollectionNameForm.svelte";
  import ConfirmDialog from "./ConfirmDialog.svelte";

  let collections = $state<Collection[]>([]);
  let loading = $state(true);
  let pending = $state(false);
  let error = $state("");
  let renaming = $state<number | null>(null);
  let deleting = $state<Collection | null>(null);
  let active = true;
  let request = 0;
  let controller: AbortController | undefined;

  async function load() {
    const token = ++request;
    controller?.abort();
    controller = new AbortController();
    loading = true;
    error = "";
    try {
      const response = await listCollections({ signal: controller.signal });
      if (active && token === request) collections = response.collections;
    } catch (err) {
      if (active && token === request && !controller.signal.aborted) error = err instanceof Error ? err.message : "Could not load collections";
    } finally {
      if (active && token === request) loading = false;
    }
  }
  $effect(() => {
    void $collectionEpoch;
    untrack(() => void load());
  });
  onMount(() => () => { active = false; controller?.abort(); });

  async function create(name: string) {
    const item = await createCollection(name);
    refreshCollections();
    if (active) collections = [item, ...collections];
  }
  async function rename(id: number, name: string) {
    const item = await renameCollection(id, name);
    refreshCollections();
    if (!active) return;
    collections = collections.map((c) => c.id === id ? item : c);
    renaming = null;
  }
  async function remove() {
    const item = deleting;
    deleting = null;
    if (!item || pending) return;
    pending = true;
    error = "";
    try {
      await deleteCollection(item.id);
      refreshCollections();
      if (active) collections = collections.filter((c) => c.id !== item.id);
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not delete collection";
    } finally {
      if (active) pending = false;
    }
  }
  const button = "px-3 py-2 border border-line rounded-lg cursor-pointer disabled:opacity-50";
</script>

<section data-collections-pane class="px-5 sm:px-9 py-5 max-w-3xl w-full">
  <p class="text-muted mb-4">Saved media in this library. Deleting a collection never deletes the original media.</p>
  <CollectionNameForm submit={create} disabled={loading || pending} />
  {#if loading}<p role="status" class="mt-4">Loading collections…</p>{/if}
  {#if error}<p role="alert" class="text-bad mt-4">{error}</p><button class={button} onclick={load} disabled={pending}>Retry</button>{/if}
  {#if !loading && !error && collections.length === 0}<p class="mt-4 text-muted">No collections yet. Create one to start saving media.</p>{/if}
  <ul class="list-none p-0 mt-5 grid gap-3">
    {#each collections as collection (collection.id)}
      <li class="p-4 bg-surface border border-line rounded-card flex flex-wrap items-center gap-2">
        {#if renaming === collection.id}
          <CollectionNameForm initial={collection.name} renaming submit={(name) => rename(collection.id, name)} />
          <button class={button} onclick={() => renaming = null}>Cancel rename</button>
        {:else}
          <button class="flex-1 text-left font-medium break-all cursor-pointer" aria-label={`Browse ${collection.name}`} onclick={() => setCollection(collection.id)}>{collection.name}</button>
          <button class={button} disabled={pending} aria-label={`Rename ${collection.name}`} onclick={() => renaming = collection.id}>Rename</button>
          <button class={button} disabled={pending} aria-label={`Delete ${collection.name}`} onclick={() => deleting = collection}>Delete collection</button>
        {/if}
      </li>
    {/each}
  </ul>
</section>
{#if deleting}
  <ConfirmDialog title="Delete collection?" detail={deleting.name} note="Only the collection and its memberships will be removed. Original images and videos stay in your library." confirmLabel="Delete collection" onconfirm={() => void remove()} oncancel={() => deleting = null} />
{/if}
