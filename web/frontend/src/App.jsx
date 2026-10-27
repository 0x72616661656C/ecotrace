import React, { useEffect, useState } from 'react';
import { Toaster, toast } from 'sonner';
import { SidebarProvider } from '@/components/ui/sidebar';
import { MapContainer } from '@/components/MapContainer';
import { AppSidebar } from '@/components/AppSidebar';

// Em produção, a interface pode apontar para uma API hospedada separadamente.
// Sem a variável, o proxy local do Vite continua usando /api.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export default function App() {
  const [selectedTiles, setSelectedTiles] = useState([]);
  const [selectedModel, setSelectedModel] = useState('unet');
  const [threshold, setThreshold] = useState(0.7);
  const [maskOpacity, setMaskOpacity] = useState(0.75);
  const [uncertaintyOpacity, setUncertaintyOpacity] = useState(0.8);
  const [useMcDropout, setUseMcDropout] = useState(false);
  const [mcSamples, setMcSamples] = useState(5);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [maskOverlayData, setMaskOverlayData] = useState(null);
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [imageCoverage, setImageCoverage] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/coverage`)
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).detail || 'Serviço indisponível');
        return response.json();
      })
      .then(setImageCoverage)
      .catch((error) => toast.error('Dados Sentinel-2 indisponíveis', { description: error.message }));
  }, []);

  const handleTileToggle = (tile) => {
    setSelectedTiles((current) => {
      const isSelected = current.some((item) => item.file === tile.file);
      if (!isSelected && current.length >= 20) {
        toast.error('Limite de 20 tiles por inferência', { description: 'Execute a análise atual ou remova um tile da seleção.' });
        return current;
      }
      const next = isSelected ? current.filter((item) => item.file !== tile.file) : [...current, tile];
      toast.info(isSelected ? 'Tile removido da seleção' : 'Tile adicionado à seleção', {
        description: `${next.length} tile${next.length === 1 ? '' : 's'} selecionado${next.length === 1 ? '' : 's'} · B2/B3/B4/B8 real`,
      });
      return next;
    });
  };

  const handleSelectModel = (model) => {
    setSelectedModel(model);
    setThreshold(model === 'attn_unet' ? 0.9 : 0.7);
  };

  const handleClearSelection = () => {
    setSelectedTiles([]);
    toast.info('Seleção limpa; as máscaras já calculadas foram mantidas.');
  };

  const handleClearMasks = () => {
    setMaskOverlayData([]);
    setResults(null);
    setShowUncertainty(false);
    toast.info('Máscaras removidas do mapa.');
  };

  const handleRunDetection = async () => {
    if (!selectedTiles.length || isProcessing) return;
    setIsProcessing(true);
    toast.loading('Processando GeoTIFF Sentinel-2 com B8/NIR real…', { id: 'detection' });

    try {
      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tile_ids: selectedTiles.map((tile) => tile.file),
          model_type: selectedModel,
          threshold,
          use_mc_dropout: useMcDropout,
          mc_samples: mcSamples,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Erro no servidor (${response.status})`);

      setResults(data);
      setShowUncertainty(data.overlays.some((overlay) => Boolean(overlay.uncertainty_png_base64)));
      setMaskOverlayData(data.overlays);
      toast.success('Análise concluída', {
        id: 'detection',
        description: `Área não-florestal estimada: ${data.stats.non_forest_pct}% · ${selectedTiles.length} tile(s)`,
      });
    } catch (error) {
      console.error('Erro na inferência Sentinel-2:', error);
      toast.error('Erro na análise', { id: 'detection', description: error.message });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <SidebarProvider>
      <div className="workspace">
        <header className="workspace-header"><div className="brand-mark">E</div><strong>EcoTrace</strong><span>Monitoramento de cobertura florestal</span></header>
        <MapContainer selectedTileIds={selectedTiles.map((tile) => tile.file)} onTileToggle={handleTileToggle} maskOverlayData={maskOverlayData} maskOpacity={maskOpacity} uncertaintyOpacity={uncertaintyOpacity} showUncertainty={showUncertainty} imageCoverage={imageCoverage} isProcessing={isProcessing} />
        <AppSidebar selectedTiles={selectedTiles} selectedModel={selectedModel} onSelectModel={handleSelectModel} threshold={threshold} onChangeThreshold={setThreshold} maskOpacity={maskOpacity} onChangeMaskOpacity={setMaskOpacity} uncertaintyOpacity={uncertaintyOpacity} onChangeUncertaintyOpacity={setUncertaintyOpacity} useMcDropout={useMcDropout} onToggleMcDropout={setUseMcDropout} mcSamples={mcSamples} onChangeMcSamples={setMcSamples} isProcessing={isProcessing} onRunDetection={handleRunDetection} onClearSelection={handleClearSelection} onClearMasks={handleClearMasks} hasMasks={maskOverlayData?.length > 0} results={results} showUncertainty={showUncertainty} onToggleUncertainty={setShowUncertainty} />
        <Toaster position="top-right" theme="light" richColors />
      </div>
    </SidebarProvider>
  );
}
