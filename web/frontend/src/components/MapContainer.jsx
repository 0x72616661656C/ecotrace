import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Layers, LocateFixed } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export function MapContainer({ selectedTileIds, onTileToggle, maskOverlayData, maskOpacity, uncertaintyOpacity, showUncertainty, imageCoverage, isProcessing }) {
  const mapElementRef = useRef(null);
  const mapRef = useRef(null);
  const overviewRef = useRef(null);
  const detailedOverlaysRef = useRef({});
  const selectedRectsRef = useRef({});
  const tilesRef = useRef([]);
  const maskRefs = useRef([]);
  const uncertaintyRefs = useRef([]);
  const liveRef = useRef({ isProcessing, onTileToggle });
  const [basemapStatus, setBasemapStatus] = useState('Carregando cobertura Sentinel-2');
  liveRef.current = { isProcessing, onTileToggle };

  useEffect(() => {
    if (mapRef.current) return undefined;
    const map = L.map(mapElementRef.current, {
      zoomControl: false,
      minZoom: 1,
      maxZoom: 20,
      maxBoundsViscosity: 1.0,
      zoomSnap: 1,
    });
    map.setView([-8, -60], 5);
    L.control.zoom({ position: 'topright' }).addTo(map);
    map.createPane('overviewPane');
    map.getPane('overviewPane').style.zIndex = 320;
    map.createPane('detailPane');
    map.getPane('detailPane').style.zIndex = 350;
    map.createPane('selectionPane');
    map.getPane('selectionPane').style.zIndex = 500;
    mapRef.current = map;
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(mapElementRef.current);
    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !imageCoverage?.tiles?.length || !imageCoverage.overall_bbox) return undefined;
    const { overall_bbox: overall, tiles } = imageCoverage;
    tilesRef.current = tiles;
    const bounds = L.latLngBounds([[overall.south, overall.west], [overall.north, overall.east]]);

    overviewRef.current?.remove();
    overviewRef.current = L.imageOverlay(`${API_BASE_URL}/overview.png`, bounds, {
      opacity: 1,
      interactive: false,
      pane: 'overviewPane',
    }).addTo(map);

    map.setMaxBounds(bounds);
    map.fitBounds(bounds, { padding: [0, 0], animate: false });
    const coverageZoom = map.getZoom();
    const minimumZoom = Math.min(20, coverageZoom + 1);
    const detailZoom = Math.min(20, minimumZoom + 1);
    map.setMinZoom(minimumZoom);
    map.setMaxZoom(Math.min(20, minimumZoom + 6));
    map.setZoom(minimumZoom, { animate: false });

    const clearDetailOverlays = () => {
      Object.values(detailedOverlaysRef.current).forEach((overlay) => overlay.remove());
      detailedOverlaysRef.current = {};
    };

    const updateDetailedOverlays = () => {
      if (map.getZoom() < detailZoom) {
        clearDetailOverlays();
        return;
      }
      const viewport = map.getBounds();
      for (const tile of tiles) {
        const visible = tile.east > viewport.getWest() && tile.west < viewport.getEast() && tile.north > viewport.getSouth() && tile.south < viewport.getNorth();
        const existing = detailedOverlaysRef.current[tile.file];
        if (visible && !existing) {
          // Sobreposição mínima evita linhas vazias por arredondamento de pixels
          // do Leaflet/Web Mercator entre GeoTIFFs contíguos.
          const longitudeBleed = (tile.east - tile.west) * 0.003;
          const latitudeBleed = (tile.north - tile.south) * 0.003;
          detailedOverlaysRef.current[tile.file] = L.imageOverlay(
            `${API_BASE_URL}/tile_image/${encodeURIComponent(tile.file)}`,
            [[tile.south - latitudeBleed, tile.west - longitudeBleed], [tile.north + latitudeBleed, tile.east + longitudeBleed]],
            { opacity: 1, interactive: false, pane: 'detailPane' },
          ).addTo(map);
        } else if (!visible && existing) {
          existing.remove();
          delete detailedOverlaysRef.current[tile.file];
        }
      }
    };

    const selectAt = (event) => {
      if (liveRef.current.isProcessing) return;
      const { lat, lng } = event.latlng;
      const tile = tiles.find((candidate) => (
        lng >= candidate.west && lng <= candidate.east && lat >= candidate.south && lat <= candidate.north
      ));
      if (!tile) return;
      liveRef.current.onTileToggle(tile);
    };

    const enforceLimits = () => {
      const nextMinimum = Math.min(20, map.getBoundsZoom(bounds, false, L.point(0, 0)) + 1);
      map.setMinZoom(nextMinimum);
      if (map.getZoom() < nextMinimum) map.setZoom(nextMinimum, { animate: false });
    };

    updateDetailedOverlays();
    map.on('moveend zoomend', updateDetailedOverlays);
    map.on('click', selectAt);
    map.on('resize', enforceLimits);
    setBasemapStatus('Mosaico Sentinel-2 · NIR real');

    return () => {
      map.off('moveend zoomend', updateDetailedOverlays);
      map.off('click', selectAt);
      map.off('resize', enforceLimits);
      clearDetailOverlays();
    };
  }, [imageCoverage]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const selected = new Set(selectedTileIds);
    for (const [file, rectangle] of Object.entries(selectedRectsRef.current)) {
      if (!selected.has(file)) {
        rectangle.remove();
        delete selectedRectsRef.current[file];
      }
    }
    for (const tile of tilesRef.current) {
      if (!selected.has(tile.file) || selectedRectsRef.current[tile.file]) continue;
      selectedRectsRef.current[tile.file] = L.rectangle([[tile.south, tile.west], [tile.north, tile.east]], {
        color: '#16a34a', weight: 2.5, opacity: 1, fillOpacity: 0.16, interactive: false, pane: 'selectionPane',
      }).addTo(map);
    }
  }, [selectedTileIds]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    maskRefs.current.forEach((overlay) => overlay.remove());
    uncertaintyRefs.current.forEach((overlay) => overlay.remove());
    maskRefs.current = [];
    uncertaintyRefs.current = [];
    if (!maskOverlayData?.length) return;

    const processedBounds = [];
    for (const overlay of maskOverlayData) {
      const bounds = [[overlay.bbox.south, overlay.bbox.west], [overlay.bbox.north, overlay.bbox.east]];
      processedBounds.push(bounds);
      maskRefs.current.push(L.imageOverlay(`data:image/png;base64,${overlay.mask_png_base64}`, bounds, { opacity: maskOpacity, interactive: false, pane: 'selectionPane' }).addTo(map));
      if (overlay.uncertainty_png_base64) {
        uncertaintyRefs.current.push(L.imageOverlay(`data:image/png;base64,${overlay.uncertainty_png_base64}`, bounds, { opacity: showUncertainty ? uncertaintyOpacity : 0, interactive: false, pane: 'selectionPane' }).addTo(map));
      }
    }
    if (processedBounds.length) {
      const combinedBounds = L.latLngBounds(processedBounds[0]);
      processedBounds.slice(1).forEach((bounds) => combinedBounds.extend(bounds));
      map.fitBounds(combinedBounds, { padding: [40, 40] });
    }
  }, [maskOverlayData]);

  useEffect(() => {
    maskRefs.current.forEach((overlay) => overlay.setOpacity(maskOpacity));
    uncertaintyRefs.current.forEach((overlay) => overlay.setOpacity(showUncertainty ? uncertaintyOpacity : 0));
  }, [maskOpacity, uncertaintyOpacity, showUncertainty]);

  const centerDataset = () => {
    const overall = imageCoverage?.overall_bbox;
    if (overall) mapRef.current?.fitBounds([[overall.south, overall.west], [overall.north, overall.east]], { padding: [0, 0] });
  };

  return (
    <main className="map-workspace" aria-label="Mapa Sentinel-2 da área de estudo">
      <div className="map-frame">
        <div id="map" ref={mapElementRef} />
        <div className="map-toolbar"><button className="map-button icon-button" aria-label="Enquadrar toda a cobertura Sentinel-2" onClick={centerDataset}><LocateFixed size={17} /></button></div>       
      </div>
    </main>
  );
}
