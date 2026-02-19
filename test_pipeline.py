"""
Script de prueba para ejecutar las funciones del DAG sin Airflow
Útil para desarrollo y debugging
"""

import sys
from pathlib import Path

# Agregar el path del DAG
sys.path.append(str(Path(__file__).parent))

from dags.bitcoin_data_pipeline import (
    descargar_datos_historicos,
    validar_datos_historicos,
    descargar_precio_actual,
    validar_precio_actual,
    generar_reporte_html,
    HISTORICAL_PATH,
    CURRENT_PATH,
    REPORTS_PATH
)
from datetime import datetime


class MockContext:
    """Mock de Airflow context para pruebas"""
    def __init__(self):
        self.xcom_data = {}
    
    class TaskInstance:
        def __init__(self, parent):
            self.parent = parent
        
        def xcom_push(self, key, value):
            self.parent.xcom_data[key] = value
            print(f"📝 XCom Push: {key} = {value}")
        
        def xcom_pull(self, task_ids, key):
            value = self.parent.xcom_data.get(key)
            print(f"📥 XCom Pull: {key} = {value}")
            return value
    
    def __getitem__(self, key):
        if key == 'task_instance':
            return self.TaskInstance(self)
        return None


def main():
    """
    Ejecuta el pipeline completo de forma secuencial
    """
    print("=" * 80)
    print("🚀 INICIANDO PRUEBA DEL PIPELINE DE BITCOIN")
    print("=" * 80)
    print()
    
    # Crear directorios
    print("📁 Creando directorios...")
    for path in [HISTORICAL_PATH, CURRENT_PATH, REPORTS_PATH]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {path}")
    print()
    
    # Mock context
    context = MockContext()
    
    # 1. Descargar datos históricos
    print("=" * 80)
    print("📥 PASO 1: Descargar datos históricos")
    print("=" * 80)
    try:
        resultado = descargar_datos_historicos(**context)
        print(f"✅ Éxito: {resultado}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    print()
    
    # 2. Validar datos históricos
    print("=" * 80)
    print("🔍 PASO 2: Validar datos históricos")
    print("=" * 80)
    try:
        resultado = validar_datos_historicos(**context)
        print(f"✅ Validación: {resultado}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    print()
    
    # 3. Descargar precio actual
    print("=" * 80)
    print("💎 PASO 3: Descargar precio actual")
    print("=" * 80)
    try:
        resultado = descargar_precio_actual(**context)
        print(f"✅ Éxito: {resultado}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    print()
    
    # 4. Validar precio actual
    print("=" * 80)
    print("🔍 PASO 4: Validar precio actual")
    print("=" * 80)
    try:
        resultado = validar_precio_actual(**context)
        print(f"✅ Validación: {resultado}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    print()
    
    # 5. Generar reporte
    print("=" * 80)
    print("📊 PASO 5: Generar reporte HTML")
    print("=" * 80)
    try:
        resultado = generar_reporte_html(**context)
        print(f"✅ Reporte generado: {resultado}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    print()
    
    # Resumen final
    print("=" * 80)
    print("✨ PIPELINE COMPLETADO CON ÉXITO")
    print("=" * 80)
    print()
    print("📁 Archivos generados:")
    print(f"   📊 Históricos: {HISTORICAL_PATH}")
    print(f"   💎 Actual: {CURRENT_PATH}")
    print(f"   📈 Reportes: {REPORTS_PATH}")
    print()
    print("🎉 ¡Todo listo! Ahora puedes:")
    print("   1. Revisar los archivos generados")
    print("   2. Abrir el reporte HTML en tu navegador")
    print("   3. Usar estos datos para entrenar tu modelo de ML")
    print()


if __name__ == "__main__":
    main()
